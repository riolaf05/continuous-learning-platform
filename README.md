### Installation

Follow those steps the first time after you clone this repository:

1. Launch Jenkins through Docker Compose:

```console
docker-compose up -d
```

If launched on cloud get the FQDN of the node where the microservices run on. If run on local machine expose ports 5000, 50000 and 8080 through **Ngrok**. 

2. Go in <node_fqdn>:8080, create a new user and create a new project while adding Github credentals and enabling GitHub GITScm hook trigger.

3. Add webhook on GitHub repository adding Ngrok address as payload URL: `<node_fqdn>/github-webhook`

4. Each new change to training or preprocess code will trigger a pipeline.

### Installation with Kubernetes
TODO

### Notes

![Figure 1](https://github.com/riolaf05/continuous-learning-platform/tree/master/img/continuous_learning_flow.jpg)

* As seen in Figure 1, the training job Dockers and the MLFLow UI Docker must share the same volume. This is because the former must write artifacts file which the latter must log into the MLFLow UI.

### TODO

* ~~Create pipeline~~
* ~~Perform continuous training~~
* ~~Trigger pipeline with Jenkins~~
* Periodically trigger a pipeline or trigger on data change

### References
* (Integrate GitHub into Jenkins)[https://www.blazemeter.com/blog/how-to-integrate-your-github-repository-to-your-jenkins-project]
* (Expose Jenkins through Ngrok)[https://dev.to/cuongld2/trigger-local-jobs-for-testing-in-jenkins-with-github-570a]
* (Install Ngrok)[https://dashboard.ngrok.com/get-started/setup]