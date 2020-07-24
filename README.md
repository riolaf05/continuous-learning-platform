### Installation

Follow those steps the first time after you clone this repository:

1. Launch Jenkins through Docker Compose:

```console
docker-compose up -d
```

2. Go in localhost:8080, create a new user and create a new project while adding Github credentals and enabling GitHub GITScm hook trigger.

3. Expose port 8080 through **Ngrok**

3. Add webhook on GitHub repository adding Ngrok address as payload URL: `<ngrok_addr>/github-webhook`

4. Each new change to training or preprocess code will trigger a pipeline.

### TODO

* Create pipeline
* Perform continuous training
* Trigger pipeline with Jenkins
* Periodically trigger a pipeline or trigger on data change

### References
* (Integrate GitHub into Jenkins)[https://www.blazemeter.com/blog/how-to-integrate-your-github-repository-to-your-jenkins-project]
* (Expose Jenkins through Ngrok)[https://dev.to/cuongld2/trigger-local-jobs-for-testing-in-jenkins-with-github-570a]
* (Install Ngrok)[https://dashboard.ngrok.com/get-started/setup]