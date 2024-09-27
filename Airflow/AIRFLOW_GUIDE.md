## Setting up Apache Airflow with Docker Compose

To set up Apache Airflow using Docker Compose, follow these steps:

### 1. Install Docker and Docker Compose

First, make sure you have Docker and Docker Compose installed on your system. You can download them from the official websites:

- Docker: https://www.docker.com/get-started
- Docker Compose: https://docs.docker.com/compose/install/

### 2. Create the project directory structure

Create a new directory for your Airflow project and navigate to it in your terminal. Then, create the following subdirectories:

```
mkdir -p airflow/{dags,logs,plugins}
```

This will create the `dags`, `logs`, and `plugins` directories inside the `airflow` directory.

### 3. Fetch the Docker Compose file

Download the latest `docker-compose.yaml` file from the Apache Airflow documentation:

```
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
```

This file contains the necessary service definitions for running Airflow with Docker Compose.

### 4. Customize the environment variables

Open the `docker-compose.yaml` file and customize the environment variables according to your needs. For example, you can change the Airflow user credentials or the PostgreSQL database password.

### 5. Place your DAGs in the dags directory

Add your Airflow DAG files to the `dags` directory you created earlier. This will make your DAGs available to the Airflow environment.

### 6. Start the Airflow environment

In the same directory as the `docker-compose.yaml` file, run the following command to start the Airflow environment:

```
docker-compose up
```

This will start the PostgreSQL database, Airflow web server, scheduler, and other necessary services.

### 7. Access the Airflow web UI

Once the environment is up and running, you can access the Airflow web UI by opening your web browser and navigating to `http://localhost:8080`. Use the default `airflow` username and password to log in.

### 8. Monitor and manage your DAGs

In the Airflow web UI, you can view, trigger, and monitor your DAGs. You can also perform various administrative tasks, such as managing connections, variables, and plugins.

## Conclusion

By following these steps, you can quickly set up a local Apache Airflow environment using Docker Compose. This approach simplifies the installation and configuration process, making it easier to get started with Airflow. Additionally, Docker Compose allows you to easily manage and scale your Airflow environment as needed.
