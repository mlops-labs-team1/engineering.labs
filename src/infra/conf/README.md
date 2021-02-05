# Infrastructure Configuration

After going through Provisioning you should have all nodes, networking and data storage created. Before
starting Configuration, you should _source_ Conda and activate your _environment_. Here are the Configuration
steps:

1. Install and activate Ansible environment
    ```sh
    $ conda env create -n my_env -f conda.yml
    $ conda activate my_env
    ```

1. Install required Collections and Roles 
    ```sh
    $ ansible-galaxy install -r requirements.yml
    ```

1. Fulfill inventory and required variables.

    - First create the file:
    ```sh
    $ mkdir group_vars
    $ touch group_vars/all.yml
    ```
    

    - After that, gather information from Terraform outputs. Check README in [prov](../prov).
    Here is a sample:
    ```yaml
    ---
    # BackEnd SQL Connection URI
    backend_uri: "postgresql://<db_user>:<db_pass>@<ip>:5432/<database>"
    
    # Google Storage URI
    artifact_uri: gs://<bucket_name>/<folder>
    
    # MLFlow Server IP
    tracking_server: <ip>
    
    # Artifact Registry that should be used to pull \ push Docker images
    artifact_registry: <gcp_registry>
    ```

1. Run Ansible playbook
    
    - ANSIBLE_HOST_KEY_CHECKING: If you want to disable SSH Host checking
    - inventory.ini: Ansible inventory with the nodes to be configured
    - --private-key: Private SSH key generated before
    - play.yml: The Playbook
    - ansible: Remote User
    - --become: Become root during the process

    ```sh
    $ ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i inventory.ini \
        --private-key=path/to/the/key/ansible-key play.yml -u ansible --become
    ```

After that, all your Infrastructure should be configured.
