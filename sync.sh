rclone copy notebook:/head-project/results results -Pv --transfers 8
rclone copy results notebook:/head-project/results -Pv --transfers 8

rclone copy notebook:/head-project/data data -Pv --transfers 8
rclone copy notebook:/head-project/checkpoint checkpoint -Pv --transfers 8
rclone copy data notebook:/head-project/data -Pv --transfers 8
rclone copy checkpoint notebook:/head-project/checkpoint -Pv --transfers 8
