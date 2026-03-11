# Numbl Execution Service

A Node.js web service for remotely executing numbl scripts.

## Features

- **Remote Execution**: Execute Numbl scripts from the web IDE on a remote server
- **Concurrency Control**: Configurable limit on concurrent executions (default: 3)
- **Timeout Protection**: Configurable execution timeout (default: 30 seconds)
- **Memory Limits**: Configurable memory limit per execution (default: 1 GB)
- **Temporary Isolation**: Each execution runs in a temporary directory

## Installation

1. Build the Numbl CLI:

```bash
npm run build:cli
```

2. Install production dependencies (if not already installed):

```bash
npm install
```

3. Build the server:

```bash
npm run build:server
```

## Configuration

Copy the example configuration file and customize as needed:

```bash
cp server/config.example.env server/.env
```

Available configuration options:

- `NUMBL_SERVICE_PORT`: Port for the service (default: 3001)
- `NUMBL_MAX_CONCURRENT`: Maximum concurrent executions (default: 3)
- `NUMBL_TIMEOUT_MS`: Execution timeout in milliseconds (default: 30000)
- `NUMBL_MAX_MEMORY_MB`: Memory limit per execution in MB (default: 1024)

## Running the Service

### Development

```bash
npm run server:dev
```

### Production

```bash
npm run server:start
```

## API Endpoints

### Health Check

```
GET /health
```

Returns the service status and current load.

**Response:**

```json
{
  "status": "ok",
  "activeExecutions": 2,
  "maxConcurrentExecutions": 3
}
```

### Get Configuration

```
GET /config
```

Returns the current service configuration.

**Response:**

```json
{
  "port": 3001,
  "maxConcurrentExecutions": 3,
  "executionTimeoutMs": 30000,
  "maxMemoryMB": 1024
}
```

### Execute Script

```
POST /execute
```

Executes a Numbl script with the provided project files.

**Request Body:**

```json
{
  "files": [
    {
      "name": "main.m",
      "content": "x = 1:10;\ndisp(x);"
    },
    {
      "name": "helper.m",
      "content": "function y = helper(x)\n  y = x * 2;\nend"
    }
  ],
  "mainScript": "main.m"
}
```

**Success Response:**

```json
{
  "success": true,
  "output": "1 2 3 4 5 6 7 8 9 10\n"
}
```

**Error Response:**

```json
{
  "success": false,
  "output": "Error: ...",
  "error": "Process exited with code 1",
  "timedOut": false
}
```

## Deployment

### Local Deployment

The service can run on the same machine as your development environment:

```bash
npm run server:start
```

### Remote Deployment (DigitalOcean, etc.)

1. Clone the repository on your server:

```bash
git clone <repository-url>
cd numbl
```

2. Install dependencies:

```bash
npm install
```

3. Build the project:

```bash
npm run build:cli
npm run build:server
```

4. Configure the service:

```bash
cp server/config.example.env server/.env
# Edit server/.env with your settings
```

5. Start the service (consider using PM2 or systemd for production):

```bash
# Using PM2
npm install -g pm2
pm2 start server/dist/execution-service.js --name numbl-service

# Or using systemd (create a service file)
sudo systemctl start numbl-service
```

### Using PM2 for Production

```bash
# Install PM2 globally
npm install -g pm2

# Start the service
pm2 start server/dist/execution-service.js --name numbl-service

# Save PM2 process list
pm2 save

# Set up PM2 to start on system boot
pm2 startup
```

## Security Considerations

⚠️ **Important**: This service executes arbitrary code. Consider the following security measures:

1. **Network Security**: Run behind a firewall, use HTTPS, implement authentication
2. **Resource Limits**: Configure appropriate timeouts and memory limits
3. **Sandboxing**: Consider running in Docker containers or VMs
4. **Rate Limiting**: Add rate limiting to prevent abuse
5. **Input Validation**: The service performs basic validation but should be enhanced for production use

## Monitoring

Check service health:

```bash
curl http://localhost:3001/health
```

View PM2 logs (if using PM2):

```bash
pm2 logs numbl-service
```

## Troubleshooting

**Service won't start:**

- Ensure the CLI is built: `npm run build:cli`
- Check if the port is already in use
- Verify Node.js version (>=18 recommended)

**Executions timing out:**

- Increase `NUMBL_TIMEOUT_MS` in configuration
- Check if scripts have infinite loops

**Memory errors:**

- Increase `NUMBL_MAX_MEMORY_MB` in configuration
- Reduce complexity of scripts being executed
