@echo off
cd /d "%~dp0"
powershell -NoExit -Command "echo 'Create input: python create_input.py';echo 'Run simulation: python Production_line_sim_current.py'; echo 'Create graphs: python graphgen_postsim.py'; echo 'Create movie: python after_movie.py'"
``