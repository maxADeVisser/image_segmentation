"""
See: http://hpc.itu.dk/scheduling/templates/local_submissions/
Script for submitting jobs to the HPC cluster at ITU.
Requires that the script is running for the entire time though.

Provide the path to the .job file you want to run as @local_job_file variable.

Requirements: 
- Python 3.x
- Paramiko library
- SCP library 
"""

import paramiko
import getpass
import time
import os
from scp import SCPClient
from dotenv import dotenv_values
from pathlib import Path

env = dotenv_values(".env")
USERNAME = env["username"]

password = getpass.getpass('Enter your password: ')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('hpc.itu.dk', username=USERNAME, password=password)

# Upload the .job file
local_job_file = <your_local_job_file.job>
remote_job_file = 'remote_job_file.job'
with SCPClient(ssh.get_transport()) as scp:
    scp.put(local_job_file, remote_job_file)

# Submit the job
stdin, stdout, stderr = ssh.exec_command(f'sbatch {remote_job_file}')
job_id = int(stdout.read().decode().split()[-1])
print(f'Submitted job with ID {job_id}')

job_finished = False
while not job_finished:
    time.sleep(5) # poll interval (seconds). Was 10 before
    stdin, stdout, stderr = ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
    output = stdout.read().decode().strip().split('\n')
    for line in output:
        if 'COMPLETED' in line:
            job_finished = True
            print(f'Job {job_id} has completed')
            break
        elif 'RUNNING' in line:
            print(f'Job {job_id} is still running...')
            break



job_output_filename = f'job.{job_id}.out'
out_directory = Path.cwd() / "dev_output"
if not out_directory.is_dir():
    out_directory.mkdir()

local_output_filename = out_directory / job_output_filename #os.path.join(out_directory, job_output_filename)

with SCPClient(ssh.get_transport()) as scp:
    scp.get(job_output_filename, local_output_filename)

print(f'Downloaded .out file to {local_output_filename}')

with open(local_output_filename) as file:
    for line in file:
        print(line)

ssh.close()