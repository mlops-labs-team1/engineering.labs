# Infrastructure Provisioning

Here follows the steps to put everything up and running. Some of them require interaction with 
Google Cloud Platform (GCP Console or  Gcloud SDK). You may find a detailed description about those 
tasks in the very Cloud documentation.

## Provisioning

1. [GCP] Enable the following GCloud APIs in Google Cloud Console

    - Compute Engine API (compute.googleapis.com)
    - Cloud Resource Manager API (cloudresourcemanager.googleapis.com)
    - Identity and Access Management API (iam.googleapis.com)
    - IAM Service Account Credentials API (iamcredentials.googleapis.com)
    - Cloud SQL Admin API (sqladmin.googleapis.com)
    - Cloud Storage API (storage.googleapis.com)
    - Cloud Run API (run.googleapis.com)

    You may use GCloud SDK as well. 

    ```sh
    gcloud services enable cloudresourcemanager.googleapis.com \
        iam.googleapis.com iamcredentials.googleapis.com sqladmin.googleapis.com \
        storage.googleapis.com run.googleapis.com compute.googleapis.com
    ```

1. [GCP] Create a terraform automation service account and set the following permissions in Console

    - Compute Admin (roles/compute.admin)
    - Compute Instance Admin v1 (roles/compute.instanceAdmin.v1)
    - Cloud SQL Admin (roles/cloudsql.admin)
    - Service Account Admin (roles/iam.serviceAccountAdmin)
    - Service Account User (roles/iam.serviceAccountUser)
    - Artifact Registry Administrator (roles/artifactregistry.admin)
    - Storage Admin (roles/storage.admin)
    - Project IAM Admin (roles/resourcemanager.projectIamAdmin)
    
    You may use GCloud to do that as well issuing the command for each role 
    `gcloud iam service-accounts create` or using some inline script. Just create a file
    with all roles you want name it as `roles.txt`and execute the script bellow.

    ```sh
    $ while read r; do
        gcloud projects add-iam-policy-binding <project_id> \
            --member="serviceAccount:terraform-automation@<project_id>.iam.gserviceaccount.com" \
            --role="$r"
      done < roles.txt
    ```

1. [GCP] Create and download the service account key using COnsole or GCloud
    
    ```sh
    $ gcloud iam service-accounts keys create path/to/key.json \
        --iam-account terraform-automation@<project_id>.iam.gserviceaccount.com
    ```

    *Note:* If you are using GCloud docker image, you should use a _workaround_. Here it is:

    ```sh
    $ docker run --rm --volumes-from gcloud-config -v path/to/dir:/tmp cloud-sdk gcloud \
        iam service-accounts keys create /tmp/key.json \
        --iam-account terraform-automation@<project_id>.iam.gserviceaccount.com
    $ sudo chown user:group path/to/dir/key.json
    ```

1. [Control Machine] Generate SSH Keys. These keys should be added to the nodes. They will be used during Configuration phase.
    ```sh
    $ ssh-keygen -t rsa -N "" -b 2048 -C "my_user" -f ./my_key
    ```

1. [Control Machine] Adjust terraform variables. All vars are declared in [vars](vars.tf) file. 
You may pass them by command line or create a `.tfvars` definition file. Terraform will load `terraform.tfvars` automatically and set the variables. Here's a sample:

    ```
    gcp_project = "myprojectid"
    
    ansible_ssh = {
    user     = "ansible"
    key_file = "/path/to/ssh_pub_key/ansible-key.pub"
    }

    mlflow_db_user = "mlflow_server_db_user"
    mlflow_db_pass = "mlflow_server_db_pass"
    ```

1. [Control Machine] This example uses a GCP Service Account credential to perform operations in the Cloud. 
Export GOOGLE_APPLICATION_CREDENTIALS env variable with the path to service account key 
(More on this [here](https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/getting_started)).

    ```
    $ export GOOGLE_APPLICATION_CREDENTIALS=/PATH_TO/SERVICE_ACCOUNT_KEY/key.json
    ```

1. [Control Machine] Intialize Terraform. Terraform will download plugins, modules and register the version for each one
of them in a `locl.hcl`file.

    ```
    $ terraform init
    ```

1. [Control Machine] Plan and Apply your Infrastructure. Terraform creates all Infra.

    ```sh
    $ terraform plan
    $ terraform apply
    ```

1. [Control Machine] Extract important information from tfstate
    
    ```sh
    # Ansible Inventory: 
    $ terraform output -raw inventory > ../conf/inventory.ini
    
    # MLFlow Server URL Connection
    $ terraform output sql_url_conn
    
    $ terraform output tracking_node
    $ terraform output training_node
    $ terraform output storage_url
    ```

## Destroying

Terraform tries to destroy the resources following a reasonable dependency order. However, you must pay
attention to resources you want to keep (Database information) or cyclical dependency (Database instance and
its user for instance). If you don't want to destroy a resource you may apply a different plan or even
remove the resource from terraform control. You can find information about that in 
[Terraform tutorials](https://learn.hashicorp.com/terraform).

This procedure destroys the managed resources:

1. [Optional] Remove DB and Buckets from Terraform management, so you can keep them for your analysis

    ```
    $ terraform state rm google_sql_user.users
    $ terraform state rm google_sql_database.database
    $ terraform state rm google_sql_database_instance.englab_db_instance
    $ terraform state rm google_storage_bucket.mlflow_bucket_name
    ```

1. Remove delete protection from Database Instance. This resource has a protection to avoid unintentional
database destruction. Just uncomment this line in [data.tf](data.tf) and apply the changes.

    ```
    # deletion_protection = false
    ```

1. Remove `google_sql_user` and `google_sql_database` from Terraform state control. There is a dependency
between database instance, database and sql user. You can't destroy the user or database meanwhile the
Database instance exists. At the same time, when you destroy the Database instance, it cascades to its
databases and users therefore causing an inconsistent state.

    ```
    $ terraform state rm google_sql_user.users
    $ terraform state rm google_sql_database.database
    ```

1. Destroy all remaining resources.

    ```
    $ terraform destroy
    ```

