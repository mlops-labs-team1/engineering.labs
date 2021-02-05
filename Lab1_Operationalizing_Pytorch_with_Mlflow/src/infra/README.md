# Infrastructure Automation

Sometimes MLOps demands a well-structured complex infrastructure. This module comprises tools
that may help with infra automation process. Here, we describe the necessary steps to create a simple
infra infrastructure allowing experimentation sessions with our "operationalisation" solution for
MLFlow. You may check our _toy example_: the [bert-classifier](../bert-classifier/) ML component.
All resources created lie in [Google Cloud Platform](https://cloud.google.com). We tried to use 
resources from free tier as much as possible, but be sure to check the resources used in `.tf files`.

## Tools & Set Up

### Terraform

[Terraform](https://www.terraform.io/) is an [IaC](https://en.wikipedia.org/wiki/Infrastructure_as_code) 
tool that helps Provisioning resources like VMs, Networking and other infra resources. It has a simple
_plan-apply-destroy_ workflow which provides for resource creation and removal from one's infrastructure
assets.

To set it up, just download the binary file from [here](https://www.terraform.io/downloads.html) and
place it in your path.

### Ansible

We need to set up resources after we create them. [Ansible](https://www.ansible.com/) is an
infrastructure automation tool that eases our tasks with infra Configuration.

It relies heavily on Python and SSH to remotely set up nodes and other elements of one's infrastructure. 
So, you must have both installed. You may install Ansible using a package from your distribution or 
from Python. You can also use [Conda](https://docs.conda.io/en/latest/). We provide a Conda environment 
file in configuration folder which you can use to install Ansible. 

```sh
$ conda env create -n <my_env> -f conda.yml
```

### Google SDK 

[GCloud SDK](https://cloud.google.com/sdk) is Google's official tool for interacting via CLI with 
its Cloud. There is a thoroughly [install guide](https://cloud.google.com/sdk/docs/install) on their 
site. We prefer using an dockerized gcloud image. So, here follows the steps to have it on and linked
to your Google Cloud Project.


```sh
$ docker pull gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# Just to ease things
$ docker tag gcr.io/google.com/cloudsdktool/cloud-sdk:slim cloud-sdk
$ docker run -ti --name gcloud-config cloud-sdk gcloud auth login
$ docker run --rm --volumes-from gcloud-config cloud-sdk gcloud config set project <project_id>

# Using alias
$ alias gcloud='docker run --rm --volumes-from gcloud-config cloud-sdk gcloud'
```

You can also drop of using Gcloud SDK and do all the initial set up in the very Google Cloud Console.


## Workflow

We have 3 stages:

- Provisioning: It's about allocating resources in your machine or in the cloud;
- Configuration: Setting up the Infra, that is, installing packages, creating users, updating 
configuration files, etc;
- Destroying: Free your local and cloud resources (or be charged).

Mostly, you're going pass through provisioning and configuration in one step. This will let your infra
up and ready to your experiments. When you're over with them, destroy your resources or be charged by your
cloud provider (remember to back up relevant data like your database). You can find detailed steps for
Provisioning and Destroying in [prov](prov/) folder. Configuration stage is better described in 
[conf](conf/).

## Resources

This is a comprehensive, yet not complete, list of resources created during Provisioning.

| Asset | Description | 
|-------|-------------|
| Training Node | VM responsible for training phase of ML Component pipeline |
| Tracking Node | VM hosting MLFlow Server |
| VPC | Virtual Network perimeter for nodes, firewall rules and other resources |
| Subnetwork | Subnetwork which define internal and external IPs for each node |
| Service Account | Service Account with proper permissions to access services and other resources in th Cloud |
| DB Instance | Postgres DB Instacnce used by MLFLow Server |
| Database | Postgres Relational datavase used by MLFLow Server |

<br>
Although we tried to fit our resources into free tier, some tasks needed more than we could find there
( for instance, DB Instances and training nodes). So, remember to double check the resources allocated 
and if they fit you budget. You may also use Google free trial plan to make some experiments.

