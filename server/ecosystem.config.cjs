module.exports = {
  apps: [
    {
      name: "numbl-service",
      script: "server/dist/execution-service.js",
      node_args: "--env-file=server/.env",
      cwd: "/home/numbl/numbl",
    },
  ],
};
