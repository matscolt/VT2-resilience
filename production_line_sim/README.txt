Production line simulator files
===============================

Files included:
- data/process_times.json
- data/transport_times.json
- data/material_stock.json
- data/bom.json
- production_line_sim.py

What changed in this version:
- The simulator now uses an explicit FIFO queue in front of every station.
- Each station can process only one phone at a time.
- Phones can arrive at a busy station, wait in that station's queue, and start only when the station becomes free.
- Transport time is applied before a phone joins the next station's queue.
- A new output folder is created for every order run, so old results are never overwritten.
- No sample output files are pre-generated anymore.
- BOM was updated so:
  - FUSE2 uses 2 Fuse
  - FUSE1 uses 1 Fuse
  - FUSE0 uses 0 Fuse
  - all variants use 1 Bottom cover, 1 Top cover, and 1 PCB

How to run in VS Code terminal:
python production_line_sim.py --order "3xFUSE2, 2xFUSE1, 4xFUSE0"

If you leave out --order, the script asks for the order in the terminal:
python production_line_sim.py

Outputs per run:
A new folder is created inside output/ for every order. Example:
output/20260413_153000__3xFUSE2_2xFUSE1_4xFUSE0/

That folder contains:
- run_metadata.json
- material_report.csv
- kpi_summary.csv
- station_schedule.csv
- unit_summary.csv
- station_summary.csv
- gantt_chart.png
- throughput_chart.png

Useful queue-related outputs:
- station_schedule.csv contains arrival_time_s, start_time_s, wait_time_s, and queue_length_on_arrival
  (number already waiting in the queue ahead of that phone, excluding the phone being processed)
- station_summary.csv contains max_queue_length, average_queue_length, average_wait_time_s,
  utilization_overall, and utilization_active_window

Utilization note:
- utilization_overall = busy time / total makespan
- utilization_active_window = busy time / (last finish at station - first start at station)

For a bottleneck like the Robot cell, utilization_active_window is often the better number to inspect,
because it shows how busy that station was once it actually started receiving work.
