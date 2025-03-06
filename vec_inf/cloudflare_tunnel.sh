#!/bin/bash

# Cloudflare tunnel setup script for LLM inference service
# This script sets up a Cloudflare tunnel for a specific job ID

# Function to check if cloudflared is installed
check_cloudflared() {
    if [ -f "${HOME}/.cloudflared/cloudflared" ]; then
        # Use the existing binary in the home directory
        export CLOUDFLARED_CMD="${HOME}/.cloudflared/cloudflared"
    elif command -v cloudflared &> /dev/null; then
        # Use the system-installed version
        export CLOUDFLARED_CMD="cloudflared"
    else
        echo "cloudflared not found, installing locally..."
        # Download cloudflared binary to the home directory
        mkdir -p "${HOME}/.cloudflared"
        curl -L --output "${HOME}/.cloudflared/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
        # Make it executable
        chmod +x "${HOME}/.cloudflared/cloudflared"
        # Use the home directory copy
        export CLOUDFLARED_CMD="${HOME}/.cloudflared/cloudflared"
    fi
}

# Function to create and start a tunnel for a specific job
create_tunnel() {
    local job_id=$1
    local hostname=$2
    local port=$3
    local subdomain=$4
    local tunnel_name="llm-inference-${job_id}"
    
    # Create tunnel config directory
    mkdir -p "${HOME}/.cloudflared/${job_id}"
    
    # Create tunnel config file
    cat > "${HOME}/.cloudflared/${job_id}/config.yml" << EOF
tunnel: ${tunnel_name}
credentials-file: ${HOME}/.cloudflared/${job_id}/credentials.json
ingress:
  - hostname: ${subdomain}.rohanmarwaha.com
    service: http://${hostname}:${port}
  - service: http_status:404
EOF
    
    # Create the tunnel and capture the output
    tunnel_create_output=$(${CLOUDFLARED_CMD} tunnel --credentials-file="${HOME}/.cloudflared/${job_id}/credentials.json" create ${tunnel_name} 2>&1)
    tunnel_create_status=$?
    
    if [ $tunnel_create_status -ne 0 ]; then
        echo "Failed to create tunnel: ${tunnel_create_output}"
        return 1
    fi
    
    # Create DNS record
    ${CLOUDFLARED_CMD} tunnel route dns ${tunnel_name} ${subdomain}.rohanmarwaha.com
    
    # Check if credentials file exists
    if [ ! -f "${HOME}/.cloudflared/${job_id}/credentials.json" ]; then
        echo "Credentials file not found. Tunnel creation may have failed."
        return 1
    fi
    
    # Start the tunnel in the background
    nohup ${CLOUDFLARED_CMD} tunnel --config="${HOME}/.cloudflared/${job_id}/config.yml" run ${tunnel_name} > "${HOME}/.cloudflared/${job_id}/tunnel.log" 2>&1 &
    
    # Save the PID for later cleanup
    echo $! > "${HOME}/.cloudflared/${job_id}/tunnel.pid"
    
    echo "Tunnel created and started for job ${job_id}"
    echo "Service available at: https://${subdomain}.rohanmarwaha.com"
    
    # Return the tunnel URL
    echo "https://${subdomain}.rohanmarwaha.com"
}

# Function to stop and delete a tunnel
delete_tunnel() {
    local job_id=$1
    local tunnel_name="llm-inference-${job_id}"
    local credentials_file="${HOME}/.cloudflared/${job_id}/credentials.json"
    
    # First, try to kill the process if we have the PID
    if [ -f "${HOME}/.cloudflared/${job_id}/tunnel.pid" ]; then
        echo "Found PID file, attempting to stop the tunnel process..."
        local pid=$(cat "${HOME}/.cloudflared/${job_id}/tunnel.pid")
        kill -TERM $pid 2>/dev/null || true
        # Give it a moment to terminate gracefully
        sleep 2
    fi
    
    # Check if credentials file exists
    if [ -f "$credentials_file" ]; then
        echo "Found credentials file, attempting to clean up and delete tunnel..."
        
        # First, get the tunnel ID/UUID
        local tunnel_id=""
        if ${CLOUDFLARED_CMD} tunnel list 2>/dev/null | grep -q "${tunnel_name}"; then
            # Extract the tunnel ID from the list output
            tunnel_id=$(${CLOUDFLARED_CMD} tunnel list | grep "${tunnel_name}" | awk '{print $1}')
            echo "Found tunnel ID: ${tunnel_id}"
        fi
        
        if [ -n "$tunnel_id" ]; then
            # Clean up active connections first
            echo "Cleaning up active connections for tunnel ${tunnel_name}..."
            ${CLOUDFLARED_CMD} tunnel cleanup ${tunnel_id} || true
            
            # Then try to delete the tunnel with force flag
            echo "Deleting tunnel ${tunnel_name}..."
            ${CLOUDFLARED_CMD} tunnel delete -f ${tunnel_id} || true
        else
            echo "Could not find tunnel ID for ${tunnel_name}"
        fi
        
        # Clean up local files
        rm -rf "${HOME}/.cloudflared/${job_id}"
    else
        # If no credentials file, try to find and delete the tunnel by name
        echo "No credentials file found, checking if tunnel exists..."
        if ${CLOUDFLARED_CMD} tunnel list 2>/dev/null | grep -q "${tunnel_name}"; then
            # Extract the tunnel ID from the list output
            local tunnel_id=$(${CLOUDFLARED_CMD} tunnel list | grep "${tunnel_name}" | awk '{print $1}')
            echo "Found tunnel ${tunnel_name} with ID ${tunnel_id}, attempting to delete..."
            
            # Clean up active connections first
            ${CLOUDFLARED_CMD} tunnel cleanup ${tunnel_id} || true
            
            # Then try to delete the tunnel with force flag
            ${CLOUDFLARED_CMD} tunnel delete -f ${tunnel_id} || true
        else
            echo "No tunnel found for job ${job_id}"
            return 0
        fi
    fi
    
    # Verify if tunnel was deleted successfully
    if ${CLOUDFLARED_CMD} tunnel list 2>/dev/null | grep -q "${tunnel_name}"; then
        echo "Warning: Failed to delete tunnel for job ${job_id}. You may need to delete it manually."
        return 1
    else
        echo "Tunnel for job ${job_id} deleted successfully"
        return 0
    fi
}

# Main function
main() {
    local action=$1
    local job_id=$2
    local hostname=$3
    local port=$4
    
    # Check if cloudflared is installed
    check_cloudflared
    
    # Login to Cloudflare if not already logged in
    if [ ! -f "${HOME}/.cloudflared/cert.pem" ]; then
        echo "Please login to Cloudflare:"
        ${CLOUDFLARED_CMD} login
    fi
    
    case $action in
        "create")
            # Use a simple subdomain based only on job ID for cleaner URLs
            local subdomain="llm-${job_id}"
            create_tunnel "$job_id" "$hostname" "$port" "$subdomain"
            ;;
        "delete")
            delete_tunnel "$job_id"
            ;;
        *)
            echo "Usage: $0 {create|delete} JOB_ID [HOSTNAME PORT]"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@" 