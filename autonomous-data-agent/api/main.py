"""
FastAPI Backend for Autonomous Data Cleaning Agent
Provides REST API endpoints for the data cleaning pipeline
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import io
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import numpy as np
import subprocess
from starlette.middleware.base import BaseHTTPMiddleware

from agents.orchestrator import AgentOrchestrator

# Custom JSON encoder for numpy/pandas types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Custom JSON Response that uses our encoder
class CustomJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(content, cls=NumpyEncoder).encode("utf-8")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous Data Cleaning Agent API",
    description="Multi-agent system for autonomous data cleaning and validation",
    version="1.0.0"
)

# Add middleware for large file uploads (500MB limit)
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size
    
    async def dispatch(self, request, call_next):
        if request.method == 'POST':
            if 'content-length' in request.headers:
                content_length = int(request.headers['content-length'])
                if content_length > self.max_upload_size:
                    return JSONResponse(
                        status_code=413,
                        content={'detail': f'File too large. Max size: {self.max_upload_size / 1024 / 1024:.0f}MB'}
                    )
        return await call_next(request)

# Apply middleware (500MB limit)
app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=524288000)

# Global state
orchestrator = None
execution_status = {
    'status': 'idle',
    'current_step': None,
    'progress': 0,
    'error': None,
    'result': None
}

# Pydantic models
class UploadResponse(BaseModel):
    message: str
    filename: str
    rows: int
    columns: int
    uploaded_at: str

class StatusResponse(BaseModel):
    status: str
    current_step: Optional[str]
    progress: int
    error: Optional[str]

class ReportResponse(BaseModel):
    quality_score: float
    verdict: str
    improvements: Dict[str, Any]
    cleaned_dataset_path: str

class RunAgentRequest(BaseModel):
    dataset_id: Optional[str] = None

# ──────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────

@app.post("/api/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset for cleaning (optimized for large files)
    
    Returns dataset info and confirmation
    """
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # For large files (>100MB), save directly without full load
        file_size = 0
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = raw_dir / f"{file.filename.replace('.csv', '')}_{timestamp}.csv"
        
        # Stream write to disk first, then read metadata
        chunk_size = 1024 * 1024  # 1MB chunks
        total_size = 0
        with open(filepath, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                total_size += len(chunk)
        
        # Read only first 10000 rows to get metadata (much faster)
        df_sample = pd.read_csv(filepath, nrows=10000)
        
        # Count total rows efficiently
        import subprocess
        try:
            if file.filename.endswith('.csv'):
                # Use wc on Windows PowerShell equivalent
                result = subprocess.run(
                    ['powershell', '-Command', f'(Get-Content "{filepath}" | Measure-Object -Line).Lines - 1'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                total_rows = int(result.stdout.strip()) if result.returncode == 0 else df_sample.shape[0]
            else:
                total_rows = df_sample.shape[0]
        except:
            total_rows = df_sample.shape[0]
        
        logger.info(f"Dataset saved: {filepath}, approx shape: {total_rows} rows, {df_sample.shape[1]} columns")
        
        execution_status['result'] = {
            'dataset_path': str(filepath),
            'shape': (total_rows, df_sample.shape[1]),
            'columns': list(df_sample.columns),
            'file_size_mb': round(total_size / (1024*1024), 2)
        }
        
        return UploadResponse(
            message="Dataset uploaded successfully",
            filename=str(filepath),
            rows=total_rows,
            columns=df_sample.shape[1],
            uploaded_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/run-agent")
async def run_agent_pipeline(request: RunAgentRequest, background_tasks: BackgroundTasks):
    """
    Run the full autonomous agent pipeline
    
    Pipeline:
    1. Profile raw data
    2. Generate cleaning strategy
    3. Execute cleaning
    4. Validate results
    5. Learn from outcomes
    
    Args:
        dataset_id: Dataset ID (optional, uses last uploaded)
        
    Returns:
        Status confirmation (actual processing happens in background)
    """
    try:
        # Get the most recent uploaded file if dataset_id not specified
        result = execution_status.get('result', {})
        dataset_path = result.get('dataset_path')
        
        if not dataset_path:
            raise HTTPException(status_code=400, detail="No dataset uploaded yet")
        
        logger.info(f"Starting agent pipeline for: {dataset_path}")
        
        # Check if file exists
        file_path = Path(dataset_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset: shape {df.shape}")
        
        # Update status
        execution_status['status'] = 'running'
        execution_status['progress'] = 0
        execution_status['error'] = None
        
        # Run pipeline in background
        background_tasks.add_task(_run_pipeline_task, df)
        
        return CustomJSONResponse({
            'status': 'started',
            'message': 'Agent pipeline started. Check /status for updates.',
            'dataset_path': dataset_path,
            'dataset_shape': list(df.shape)
        })
    
    except Exception as e:
        logger.error(f"Pipeline start error: {str(e)}")
        execution_status['status'] = 'error'
        execution_status['error'] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """
    Get current pipeline execution status
    
    Returns:
        Current status, step, and progress (0-100)
    """
    return StatusResponse(
        status=execution_status['status'],
        current_step=execution_status['current_step'],
        progress=execution_status['progress'],
        error=execution_status['error']
    )

@app.get("/api/report")
async def get_report():
    """
    Get the full cleaning report after execution
    
    Returns:
        - Data quality report (JSON)
        - Agent reasoning logs
        - Quality score and verdict
    """
    try:
        if execution_status['status'] != 'completed':
            raise HTTPException(
                status_code=400, 
                detail=f"Pipeline not completed. Current status: {execution_status['status']}"
            )
        
        result = execution_status['result']
        quality_metrics = result.get('quality_metrics', {})
        original_shape = result.get('original_shape', (0, 0))
        cleaned_shape = result.get('cleaned_shape', (0, 0))
        
        # Convert numpy types to native Python types
        def convert_to_native(value):
            if isinstance(value, (np.integer, np.floating)):
                return int(value) if isinstance(value, np.integer) else float(value)
            return value
        
        # Transform metrics to match frontend expectations
        transformed_metrics = {
            'cleaned_quality_score': convert_to_native(quality_metrics.get('cleaned_quality_score', 0)),
            'quality_improvement': convert_to_native(quality_metrics.get('quality_improvement_points', 0)),
            'data_retention_pct': convert_to_native(quality_metrics.get('data_retention_pct', 0)),
            'completeness_pct': convert_to_native(quality_metrics.get('cleaned_completeness_pct', 0)),
            'original_rows': int(original_shape[0]),
            'cleaned_rows': int(cleaned_shape[0])
        }
        
        profiling_report = result.get('reports', {}).get('profiling', {})
        strategy_report = result.get('reports', {}).get('strategy', {})
        transformation_log = result.get('reports', {}).get('transformation_log', {})
        
        return CustomJSONResponse(
            {
                'status': 'success',
                'validation_verdict': result.get('verdict', 'PASS'),
                'verdict': result.get('verdict', 'PASS'),
                'quality_metrics': transformed_metrics,
                'decision_summary': {
                    'profiling_decisions': len(profiling_report.get('profiles', [])) if isinstance(profiling_report.get('profiles'), list) else 0,
                    'strategy_decisions': len(strategy_report.get('actions', [])) if isinstance(strategy_report.get('actions'), list) else 0,
                    'transformations_applied': len(transformation_log) if isinstance(transformation_log, dict) else 0,
                    'quality_improvement': convert_to_native(quality_metrics.get('quality_improvement_points', 0))
                },
                'shape_improvement': {
                    'original': list(original_shape),
                    'cleaned': list(cleaned_shape)
                },
                'reports': {
                    'profiling': profiling_report,
                    'validation': result.get('reports', {}).get('validation'),
                    'learning': result.get('reports', {}).get('learning')
                },
                'cleaned_dataset_path': result.get('cleaned_dataset_path'),
                'agent_logs': {
                    'profiling': result.get('agent_logs', {}).get('profiling'),
                    'strategy': result.get('agent_logs', {}).get('strategy'),
                    'validation': result.get('agent_logs', {}).get('validation')
                }
            },
            media_type="application/json"
        )
    
    except Exception as e:
        logger.error(f"Report error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-cleaned-data")
async def download_cleaned_data():
    """
    Download the cleaned dataset as CSV
    """
    try:
        if execution_status['result'] is None:
            raise HTTPException(status_code=400, detail="No cleaned data available")
        
        filepath = execution_status['result'].get('cleaned_dataset_path')
        if not filepath or not Path(filepath).exists():
            raise HTTPException(status_code=404, detail="Cleaned data not found")
        
        return FileResponse(
            path=filepath,
            filename="cleaned_data.csv",
            media_type="text/csv"
        )
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-report")
async def download_report():
    """
    Download the validation report as JSON
    """
    try:
        result = execution_status.get('result', {})
        
        # Find the validation report file
        reports_dir = Path('data/reports')
        if reports_dir.exists():
            report_files = list(reports_dir.glob('validation_report_*.json'))
            if report_files:
                latest_report = sorted(report_files)[-1]
                return FileResponse(
                    path=latest_report,
                    filename="validation_report.json",
                    media_type="application/json"
                )
        
        raise HTTPException(status_code=404, detail="Report not found")
    
    except Exception as e:
        logger.error(f"Report download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return CustomJSONResponse({
        'status': 'healthy',
        'service': 'Autonomous Data Cleaning Agent API',
        'version': '1.0.0'
    })

# ──────────────────────────────────────────────────────────
# BACKGROUND TASKS
# ──────────────────────────────────────────────────────────

def _run_pipeline_task(df: pd.DataFrame):
    """Background task to run the agent pipeline (optimized)"""
    global execution_status
    
    try:
        # For very large DataFrames (>500MB), use aggressive sampling
        file_size_estimate = df.memory_usage(deep=True).sum() / (1024**3)
        
        if file_size_estimate > 0.3:  # >300MB
            logger.info(f"Large dataset detected ({file_size_estimate:.2f}GB). Using fast-track processing...")
            sample_fraction = min(0.3, 100000 / len(df)) if len(df) > 100000 else 1.0
            df_process = df.sample(frac=sample_fraction, random_state=42)
            logger.info(f"Processing {len(df_process):,} rows ({sample_fraction*100:.1f}% of {len(df):,})")
        else:
            df_process = df
        
        orchestrator = AgentOrchestrator()
        
        # Run pipeline
        result = orchestrator.run_pipeline(df_process)
        
        # Save results
        orchestrator.save_results(result)
        
        # Save cleaned data
        if result['cleaned_df'] is not None:
            output_dir = Path('data/cleaned')
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cleaned_path = output_dir / f"cleaned_{timestamp}.csv"
            result['cleaned_df'].to_csv(cleaned_path, index=False)
            result['cleaned_dataset_path'] = str(cleaned_path)
        
        # Update status
        execution_status['status'] = 'completed'
        execution_status['progress'] = 100
        execution_status['current_step'] = 'Pipeline completed'
        execution_status['result'] = result
        
        logger.info("Pipeline execution completed successfully")
    
    except Exception as e:
        logger.error(f"Pipeline execution error: {str(e)}")
        execution_status['status'] = 'error'
        execution_status['error'] = str(e)
        execution_status['progress'] = 0

# ──────────────────────────────────────────────────────────
# STATIC FILES & DOCUMENTATION
# ──────────────────────────────────────────────────────────
# Note: Static files are mounted at "/" and serve index.html automatically

@app.get("/docs-ui", response_class=HTMLResponse)
async def api_docs():
    """API documentation link"""
    return """
    <html>
        <head><title>API Docs</title></head>
        <body>
            <h2>API Endpoints</h2>
            <p><a href="/api/health">Health Check</a></p>
            <p>See <a href="/docs">Swagger UI</a> for interactive API docs.</p>
        </body>
    </html>
    """

# ──────────────────────────────────────────────────────────
# MOUNT STATIC FILES (at the end, after all API routes)
# ──────────────────────────────────────────────────────────
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

# ──────────────────────────────────────────────────────────
# STARTUP/SHUTDOWN
# ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Autonomous Data Cleaning Agent API starting...")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    logger.info("✓ Directories initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Autonomous Data Cleaning Agent API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        limit_max_requests=10000,
        limit_upload_size=524288000  # 500MB in bytes
    )
