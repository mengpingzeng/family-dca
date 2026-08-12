import os, sys, subprocess
os.chdir('/mnt/data/zmp/family-dca')
# Double-fork to fully detach
if os.fork() != 0:
    sys.exit(0)
os.setsid()
if os.fork() != 0:
    sys.exit(0)
os.chdir('/')
log = open('/tmp/uvicorn.log', 'a')
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
os.execvp(sys.executable, [sys.executable, '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000'])
