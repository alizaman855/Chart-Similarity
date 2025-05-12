import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
from pathlib import Path
from typing import Optional

# Import chart similarity analysis functions from your core module
from chart_similarity import find_most_similar_charts_in_video, prepare_results_for_json

# Create FastAPI app
app = FastAPI(title="Chart Similarity Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and results
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
STATIC_DIR = Path("static")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Create category directories
for category in ["gold", "btc", "usdcad"]:
    (RESULTS_DIR / category).mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Store job status in memory
jobs = {}

class AnalysisParams(BaseModel):
    fps: float = 1.0
    detect_color: str = "green"
    category: Optional[str] = None
    

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Return the main HTML file"""
    return FileResponse("static/index.html")


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    category: str = Form(None)
):
    """Handle video upload with category"""
    # Validate video file
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    
    # Validate category
    if category not in ["gold", "btc", "usdcad", None]:
        raise HTTPException(status_code=400, detail="Invalid category. Must be 'gold', 'btc', or 'usdcad'.")
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Create job directory in uploads
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = job_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job entry
    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "filename": file.filename,
        "file_path": str(file_path),
        "created_at": time.time(),
        "results": None,
        "progress": 0,
        "category": category
    }
    
    return {"job_id": job_id, "filename": file.filename, "status": "uploaded", "category": category}


@app.post("/api/analyze/{job_id}")
async def analyze_video(job_id: str, background_tasks: BackgroundTasks, params: AnalysisParams):
    """Start video analysis in background"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Update job status
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["params"] = params.dict()
    jobs[job_id]["progress"] = 0
    
    # If category was provided in params, update job category
    if params.category:
        jobs[job_id]["category"] = params.category
    
    # Create results directory
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(exist_ok=True)
    
    # Run analysis in background
    background_tasks.add_task(
        run_analysis, 
        job_id=job_id,
        file_path=jobs[job_id]["file_path"],
        output_dir=str(result_dir),
        fps=params.fps,
        category=jobs[job_id].get("category")
    )
    
    return {"job_id": job_id, "status": "processing", "category": jobs[job_id].get("category")}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    response = {
        "id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "progress": job.get("progress", 0),
        "category": job.get("category")
    }
    
    # Add results if job is completed
    if job["status"] == "completed" and job.get("results"):
        # Add category to results if available
        if "category" in job and job["category"]:
            if isinstance(job["results"], dict):
                job["results"]["category"] = job["category"]
        
        response["results"] = job["results"]
    
    # Add error if job failed
    if job["status"] == "failed" and "error" in job:
        response["error"] = job["error"]
    
    return response


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    job_list = []
    for job_id, job in jobs.items():
        job_list.append({
            "id": job["id"],
            "status": job["status"],
            "filename": job["filename"],
            "created_at": job["created_at"],
            "category": job.get("category")
        })
    
    # Sort by created_at (newest first)
    job_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"jobs": job_list}


def update_progress(job_id, progress):
    """Update job progress"""
    if job_id in jobs:
        jobs[job_id]["progress"] = progress


def run_analysis(job_id: str, file_path: str, output_dir: str, fps: float, category: str = None):
    """Run video analysis in background"""
    try:
        # Progress callback function
        def progress_callback(progress):
            update_progress(job_id, progress)
        
        # Run analysis
        results = find_most_similar_charts_in_video(
            video_path=file_path,
            output_dir=output_dir,
            fps=fps,
            progress_callback=progress_callback
        )
        
        # Prepare results for JSON
        serializable_results = prepare_results_for_json(results)
        
        # Add category to results
        if category:
            serializable_results["category"] = category
        
        # Save results to file
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(serializable_results, f)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = serializable_results
        jobs[job_id]["progress"] = 100
        
    except Exception as e:
        # Update job status with error
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Error processing job {job_id}: {str(e)}")


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove from memory
    job = jobs.pop(job_id)
    
    # Remove files
    job_upload_dir = UPLOAD_DIR / job_id
    job_result_dir = RESULTS_DIR / job_id
    
    if job_upload_dir.exists():
        shutil.rmtree(job_upload_dir)
    
    if job_result_dir.exists():
        shutil.rmtree(job_result_dir)
    
    return {"status": "deleted", "job_id": job_id}


@app.get("/api/jobs/category/{category}")
async def list_jobs_by_category(category: str):
    """List all jobs for a specific category"""
    if category not in ["gold", "btc", "usdcad", "all"]:
        raise HTTPException(status_code=400, detail="Invalid category. Must be 'gold', 'btc', 'usdcad', or 'all'.")
    
    job_list = []
    for job_id, job in jobs.items():
        if category == "all" or job.get("category") == category:
            job_list.append({
                "id": job["id"],
                "status": job["status"],
                "filename": job["filename"],
                "created_at": job["created_at"],
                "category": job.get("category")
            })
    
    # Sort by created_at (newest first)
    job_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"jobs": job_list}


@app.get("/api/cleanup")
async def cleanup_old_jobs():
    """Cleanup old jobs (older than 24 hours)"""
    current_time = time.time()
    removed_count = 0
    
    # Find jobs to remove
    jobs_to_remove = []
    for job_id, job in jobs.items():
        if current_time - job["created_at"] > 24 * 60 * 60:  # 24 hours
            jobs_to_remove.append(job_id)
    
    # Remove jobs
    for job_id in jobs_to_remove:
        # Remove from memory
        if job_id in jobs:
            del jobs[job_id]
        
        # Remove files
        job_upload_dir = UPLOAD_DIR / job_id
        job_result_dir = RESULTS_DIR / job_id
        
        if job_upload_dir.exists():
            shutil.rmtree(job_upload_dir)
        
        if job_result_dir.exists():
            shutil.rmtree(job_result_dir)
            
        removed_count += 1
    
    return {"removed": removed_count}


if __name__ == "__main__":
    print("Starting Chart Similarity Analyzer API...")
    print("Open http://localhost:5000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=5000)