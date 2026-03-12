import os 
from abc import ABC, abstractmethod
import subprocess
import re

class scheduler(ABC):

    @abstractmethod
    def submit(self,jobscript):
        '''Submit a job'''
        pass 

    @abstractmethod
    def chain_submission(self,jobscript,jobid):
        '''Chain jobs'''
        pass 

    @abstractmethod
    def cancel(self,jobid):
        '''Cancel a job'''
        pass 
    
    @abstractmethod
    def getstatus(self,jobid):
        '''Get job status'''
        pass 
    

class slurm_scheduler(scheduler):

    def submit(self,jobscript):
        result = subprocess.run(
            ["sbatch", jobscript],
            capture_output=True,
            text=True
        )
        # Check for failure
        if result.returncode != 0:
            raise RuntimeError(
                f"Submission failed (code {result.returncode}):\n{result.stderr}"
            )
        jobid = self._get_jobid(result)
        return jobid 
    
    def chain_submission(self, jobscript, jobid):
        # Build sbatch command with dependency
        cmd = ["sbatch", f"--dependency=afterok:{jobid}", jobscript]

        # Submit and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check for failure
        if result.returncode != 0:
            raise RuntimeError(
                f"Chained submission failed (code {result.returncode}):\n{result.stderr}"
            )
        subjid =  self._get_jobid(result)
        return subjid 


    def cancel(self, jobid):
        """Cancel a SLURM job"""
        result = subprocess.run(
            ["scancel", str(jobid)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to cancel job {jobid}:\n{result.stderr}"
            )

    def getstatus(self, jobid):
        """Return SLURM job status as a string"""
        result = subprocess.run(
            ["squeue", "--job", str(jobid), "--noheader", "--format=%T"],
            capture_output=True,
            text=True
        )
        status = result.stdout.strip()
        if not status:
            return "COMPLETED_OR_UNKNOWN"  # job not in queue
        return status

    def _get_jobid(self, output):
        out = output.stdout if hasattr(output, "stdout") else ""
        match = re.search(r"Submitted batch job (\d+)", out)
        if match:
            return int(match.group(1))
        else:
            raise RuntimeError(
                f"Job submission failed. sbatch output:\n{out}\nReturn code: {output.returncode}"
            )
    