#!/bin/bash

measure() {
	command_to_run="$@"

	# Check if nvidia-smi is available (for NVIDIA GPUs)
	if command -v nvidia-smi &>/dev/null; then
		has_nvidia=true
	else
		has_nvidia=false
		echo "Note: nvidia-smi not found. GPU monitoring will be disabled."
	fi

	# Get CPU info
	cpu_mhz=$(grep -m 1 "cpu MHz" /proc/cpuinfo | awk '{print $4}')
	cpu_cores=$(nproc)
	total_cpu_mhz=$(echo "$cpu_mhz * $cpu_cores" | bc)

	# Record start time
	start_time=$(date +%s.%N)

	# Run the command
	eval "$command_to_run" &
	cmd_pid=$!

	# Initialize peak tracking
	peak_cpu_percent=0
	peak_mem_kb=0
	peak_gpu_percent=0
	peak_gpu_mem_mb=0

	# Create temporary files for GPU monitoring
	if $has_nvidia; then
		gpu_util_file=$(mktemp)
		gpu_mem_file=$(mktemp)
	fi

	# Monitor usage while command runs
	while ps -p $cmd_pid >/dev/null 2>&1; do
		# Get all child PIDs including the parent
		pids=$(pstree -p $cmd_pid | grep -o '([0-9]\+)' | tr -d '()' | tr '\n' ' ')$cmd_pid

		# Get current total CPU usage percentage across all processes
		current_cpu=0
		for pid in $pids; do
			if ps -p $pid >/dev/null 2>&1; then
				pid_cpu=$(ps -p $pid -o %cpu= | tr -d ' ')
				current_cpu=$(echo "$current_cpu + $pid_cpu" | bc 2>/dev/null || echo "$current_cpu")
			fi
		done

		# Get current total memory usage in KB across all processes
		current_mem=0
		for pid in $pids; do
			if ps -p $pid >/dev/null 2>&1; then
				pid_mem=$(ps -p $pid -o rss= | tr -d ' ')
				current_mem=$(echo "$current_mem + $pid_mem" | bc 2>/dev/null || echo "$current_mem")
			fi
		done

		# Get GPU usage if available
		if $has_nvidia; then
			# Get GPU utilization percentage
			gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
			echo "$gpu_util" >>"$gpu_util_file"

			# Get GPU memory usage in MB
			gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
			echo "$gpu_mem" >>"$gpu_mem_file"
		fi

		# Update peaks if current is higher
		if [[ -n "$current_cpu" ]] && (($(echo "$current_cpu > $peak_cpu_percent" | bc -l 2>/dev/null))); then
			peak_cpu_percent=$current_cpu
		fi

		if [[ -n "$current_mem" ]] && ((current_mem > peak_mem_kb)); then
			peak_mem_kb=$current_mem
		fi

		sleep 0.1 # Faster sampling rate
	done

	# Calculate total runtime
	end_time=$(date +%s.%N)
	total_time=$(echo "scale=3; $end_time - $start_time" | bc -l)

	# Get peak GPU values if available
	if $has_nvidia; then
		peak_gpu_percent=$(sort -nr "$gpu_util_file" | head -1)
		peak_gpu_mem_mb=$(sort -nr "$gpu_mem_file" | head -1)
		rm -f "$gpu_util_file" "$gpu_mem_file"
	fi

	# Convert peak CPU percentage to absolute CPU usage
	peak_cpu_mhz=$(echo "scale=2; $peak_cpu_percent * $total_cpu_mhz / 100" | bc)

	# Format CPU usage in appropriate units (MHz, GHz)
	if (($(echo "$peak_cpu_mhz >= 1000" | bc -l))); then
		peak_cpu_ghz=$(echo "scale=2; $peak_cpu_mhz / 1000" | bc)
		cpu_unit="${peak_cpu_ghz}G"
	else
		cpu_unit="${peak_cpu_mhz}M"
	fi

	# Format memory usage in appropriate units (KB, MB, GB)
	if ((peak_mem_kb >= 1048576)); then
		peak_mem_gb=$(echo "scale=2; $peak_mem_kb / 1048576" | bc)
		mem_unit="${peak_mem_gb}G"
	elif ((peak_mem_kb >= 1024)); then
		peak_mem_mb=$(echo "scale=2; $peak_mem_kb / 1024" | bc)
		mem_unit="${peak_mem_mb}M"
	else
		mem_unit="${peak_mem_kb}K"
	fi

	# Format GPU memory usage
	if $has_nvidia; then
		if ((peak_gpu_mem_mb >= 1024)); then
			peak_gpu_mem_gb=$(echo "scale=2; $peak_gpu_mem_mb / 1024" | bc)
			gpu_mem_unit="${peak_gpu_mem_gb}G"
		else
			gpu_mem_unit="${peak_gpu_mem_mb}M"
		fi
	fi

	# Output results in a simple format
	echo "runtime: $total_time s"
	echo "peak_cpu: $cpu_unit"
	echo "peak_mem: $mem_unit"

	if $has_nvidia; then
		echo "peak_gpu: ${peak_gpu_percent}%"
		echo "peak_gpu_mem: $gpu_mem_unit"
	fi
}

# Check if arguments were provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 command [arguments]"
	exit 1
fi

# Call the measure function with all arguments
measure "$@"
