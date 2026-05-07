@echo off
cd /d "%~dp0"
powershell -NoExit -Command "echo 'Create input: python A_input.py';echo 'Run simulation: python D_production_line_sim.py'; echo 'Create graphs: python F_graphgen_postsim.py'; echo 'Create movie: python G_after_movie.py'; echo 'Run the entire script: python _MAIN.py';"
`