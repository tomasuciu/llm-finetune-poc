# Name of the company. It is used for context name of the cluster in .kubeconfig file.
company_name = "poc"

# Whether to store the controller state on filestore or network SSD.
controller_state_on_filestore = false

# Shared filesystem to be used on controller nodes.
# Deprecated: Starting with version 1.22, this variable isn't used, as controller state is stored on network SSD disks.
# Remains for the backward compatibility.
# ---
filestore_controller_spool = {
  spec = {
    size_gibibytes       = 93
    block_size_kibibytes = 4
  }
}

# Shared filesystem to be used on controller, worker, and login nodes.
# Notice that auto-backups are enabled for filesystems with size less than 12 TiB.
# If you need backups for jail larger than 12 TiB, set 'backups_enabled' to 'force_enable' down below.
# ---
 filestore_jail = {
   spec = {
     size_gibibytes       = 186
     block_size_kibibytes = 4
   }
 }

# Additional shared filesystems to be mounted inside jail.
# If a big filesystem is needed it's better to deploy this additional storage because jails bigger than 12 TiB
# ARE NOT BACKED UP by default.
# ---
filestore_jail_submounts = [
    {
        name       = "training-data"
        mount_path = "/mnt/training-data"
        spec = {
            size_gibibytes       = 1024
            block_size_kibibytes = 4
        }
    },
    { 
        name       = "checkpoints"
        mount_path = "/mnt/checkpoints"
        spec = {
            size_gibibytes       = 745
            block_size_kibibytes = 4
        }
    }
]

# Additional (Optional) node-local Network-SSD disks to be mounted inside jail on worker nodes.
# It will create compute disks with provided spec for each node via CSI.
# NOTE: in case of `NETWORK_SSD_NON_REPLICATED` disk type, `size` must be divisible by 93Gi - https://docs.nebius.com/compute/storage/types#disks-types.

node_local_jail_submounts = [{
  name            = "scratch"
  mount_path      = "/mnt/scratch"
  size_gibibytes  = 186
  disk_type       = "NETWORK_SSD"
  filesystem_type = "xfs"
}]

# Whether to create extra NRD disks for storing Docker/Enroot images and container filesystems on each worker node.
# It will create compute disks with provided spec for each node via CSI.
# NOTE: In case you're not going to use Docker/Enroot in your workloads, it's worth disabling this feature.
# NOTE: `size` must be divisible by 93Gi - https://docs.nebius.com/compute/storage/types#disks-types.
# ---
node_local_image_disk = {
  enabled = false
}
# ---

#node_local_image_disk = {
#  enabled = true
#  spec = {
#    size_gibibytes  = 930
#    filesystem_type = "ext4"
#    # Could be changed to `NETWORK_SSD_NON_REPLICATED`
#    disk_type = "NETWORK_SSD_IO_M3"
#  }
#}

# Shared filesystem to be used for accounting DB.
# By default, null.
# Required if accounting_enabled is true.
# ---
#filestore_accounting = {
#  #spec = {
#  #  size_gibibytes       = 93
#  #  block_size_kibibytes = 4
#  #}
#}
# Or use existing filestore.
# ---
# filestore_accounting = {
#   existing = {
#     id = "computefilesystem-<YOUR-FILESTORE-ID>"
#   }
# }

# endregion Storage

# region nfs-server

# nfs = {
#   enabled        = false
#   size_gibibytes = 3720
#   mount_path     = "/home"
#   resource = {
#     platform = "cpu-d3"
#     preset   = "32vcpu-128gb"
#   }
#   public_ip = false
# }

nfs_in_k8s = {
  enabled        = false
  #size_gibibytes = 2046 # 93 * 22
}

# endregion nfs-server

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                                                                                      #
#                                                         Slurm                                                        #
#                                                                                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Slurm

# Version of soperator.
# ---
#slurm_operator_version = "1.22.1"
slurm_operator_version = "1.22.0"

# Is the version of soperator stable or not.
# ---
slurm_operator_stable = true

# Type of the Slurm partition config. Could be either `default` or `custom`.
# By default, "default".
# ---
slurm_partition_config_type = "default"

# Partition config in case of `custom` slurm_partition_config_type.
# Each string must be started with `PartitionName`.
# By default, empty list.
# ---
# slurm_partition_raw_config = [
#   "PartitionName=low_priority Nodes=low_priority Default=YES MaxTime=INFINITE State=UP PriorityTier=1",
#   "PartitionName=high_priority Nodes=low_priority Default=NO MaxTime=INFINITE State=UP PriorityTier=2"
# ]
# If Nodes present, they must not contain node names: use only nodeset values, "ALL" or "".
# If nodesets are used in the partition config, slurm_worker_features with non-empty nodeset_name
# must be declared (see below).
# Specifying specific nodes is not supported since Dynamic Nodes are used.
# For more details, see https://slurm.schedmd.com/dynamic_nodes.html#partitions.

# List of features to be enabled on worker nodes. Each feature object has:
# - name: (Required) The name of the feature.
# - hostlist_expr: (Required) A Slurm hostlist expression, e.g. "workers-[0-2,10],workers-[3-5]".
#   Soperator will run these workers with the feature name.
# - nodeset_name: (Optional) The Slurm nodeset name to be provisioned using this feature.
#   This nodeset may be used in conjunction with partitions.
#   It is required if `Nodes=<nodeset_name>` is used for a partition.
#
# slurm_worker_features = [
#   {
#     name = "low_priority"
#     hostlist_expr = "worker-[0-0]"
#     nodeset_name = "low_priority"
#   },
#   {
#     name = "low_priority"
#     hostlist_expr = "worker-1"
#     nodeset_name = "high_priority"
#   }
# ]

# Health check config:
# - health_check_interval: (Required) Interval for health check run in seconds.
# - health_check_program: (Required) Program for health check run.
# - health_check_node_state: (Required) What node states should execute the program.
#
# slurm_health_check_config = {
#   health_check_interval: 30,
#   health_check_program: "/usr/bin/gpu_healthcheck.sh",
#   health_check_node_state: [
#     {
#       state: "ANY"
#     },
#     {
#       state: "CYCLE"
#     }
#   ]
# }

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                         Nodes                                                        #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Nodes

# Configuration of System node set for system resources created by Soperator.
# Keep in mind that the k8s nodegroup will have auto-scaling enabled and the actual number of nodes depends on the size
# of the cluster.
# ---
slurm_nodeset_system = {
  min_size = 3
  max_size = 9
  resource = {
    platform = "cpu-d3"
    preset   = "8vcpu-32gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }
}

# Configuration of Slurm Controller node set.
# ---
slurm_nodeset_controller = {
  size = 2
  resource = {
    platform = "cpu-d3"
    preset   = "8vcpu-32gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }
}

# Configuration of Slurm Worker node sets.
# There can be only one Worker node set for a while.
# nodes_per_nodegroup allows you to split node set into equally-sized node groups to keep your cluster accessible and working
# during maintenance. Example: nodes_per_nodegroup=3 for size=12 nodes will create 4 groups with 3 nodes in every group.
# infiniband_fabric is required field
# ---
slurm_nodeset_workers = [{
  size                    = 4
  nodes_per_nodegroup     = 4
  max_unavailable_percent = 50
  # max_surge_percent       = 50
  # drain_timeout           = "10s"
  resource = {
    platform = "gpu-h200-sxm"
    preset   = "1gpu-16vcpu-200gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 512
    block_size_kibibytes = 4
  }
  gpu_cluster = {
    infiniband_fabric = "fabric-7"
  }
  # Change to preemptible = {} in case you want to use preemptible nodes
  preemptible = null
}]

# Driverfull mode is used to run Slurm jobs with GPU drivers installed on the worker nodes.
use_preinstalled_gpu_drivers = true

# Configuration of Slurm Login node set.
# ---
slurm_nodeset_login = {
  size = 2
  resource = {
    platform = "cpu-d3"
    preset   = "32vcpu-128gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 256
    block_size_kibibytes = 4
  }
}

# Configuration of Slurm Accounting node set.
# Required in case of Accounting usage.
# By default, null.
# ---
#slurm_nodeset_accounting = {
#  resource = {
#    platform = "cpu-d3"
#    preset   = "8vcpu-32gb"
#  }
#  boot_disk = {
#    type                 = "NETWORK_SSD"
#    size_gibibytes       = 128
#    block_size_kibibytes = 4
#  }
#}

#----------------------------------------------------------------------------------------------------------------------#
#                                                         Login                                                        #
#----------------------------------------------------------------------------------------------------------------------#
# region Login

# Public or private ip for login node load balancer
# By default, true (public).
# ---
slurm_login_public_ip = true

# Whether to enable Tailscale init container on login pod.
# By default, false
# ---
tailscale_enabled = false

# Authorized keys accepted for connecting to Slurm login nodes via SSH as 'root' user.
# ---
slurm_login_ssh_root_public_keys = [
  "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPPZRTfsp2sC8AfgxYRgCRrfpu33OHyC4+1+67WpCX54 tomasuciu@gmail.com",
]

# endregion Login

#----------------------------------------------------------------------------------------------------------------------#
#                                                       Exporter                                                       #
#----------------------------------------------------------------------------------------------------------------------#
# region Exporter

# Whether to enable Slurm metrics exporter.
# By default, true.
# ---
slurm_exporter_enabled = true

# endregion Exporter

#----------------------------------------------------------------------------------------------------------------------#
#                                                      ActiveChecks                                                    #
#----------------------------------------------------------------------------------------------------------------------#
# region ActiveChecks

# Scope of active checks. Defines what active checks should be checked during cluster bootstrap.
# By default, prod.
# All values: prod, dev, testing.
# Defaults of the chart: https://github.com/nebius/soperator/blob/1a8e7e322a3dc84974b4f25890e26f8e19c20eb6/helm/soperator-activechecks/values.yaml#L28
# Defaults override: https://github.com/nebius/nebius-solutions-library/blob/9e971de4d85aeb2799e71a163ed47c8480878314/soperator/modules/slurm/locals_active_checks.tf
# ---
active_checks_scope = "dev"

# endregion ActiveChecks

# endregion Nodes

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                        Config                                                        #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Config

# Shared memory size for Slurm controller and worker nodes in GiB.
# By default, 64.
# ---
slurm_shared_memory_size_gibibytes = 1024

# endregion Config
#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                       Telemetry                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Telemetry

# Whether to enable telemetry.
# By default, true.
# ---
telemetry_enabled = true

# Whether to enable dcgm job mapping (adds hpc_job label on DCGM_ metrics).
# By default, true.
# ---
dcgm_job_mapping_enabled = true

# Configuration of the Soperator Notifier (https://github.com/nebius/soperator/tree/main/helm/soperator-notifier).
# ---
# soperator_notifier = {
#   enabled           = true
#   slack_webhook_url = "https://hooks.slack.com/services/X/Y/Z"
# }
soperator_notifier = {
  enabled = false
}

#public_o11y_enabled = true
public_o11y_enabled = false

# endregion Telemetry

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                       Accounting                                                     #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Accounting

# Whether to enable Accounting.
# By default, true.
# ---
accounting_enabled = false

# endregion Accounting

# endregion Slurm

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                       Backups                                                        #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Backups

# Whether to enable Backups. Choose from 'auto', 'force_enable', 'force_disable'.
# 'auto' turns backups on for jails with max size less than 12 TB and is a default option.
# ---
#backups_enabled = "auto"
backups_enabled = "force_disable"

# Password to be used for encrypting jail backups.
# ---
backups_password = "password"

# Cron schedule for backup task.
# See https://docs.k8up.io/k8up/references/schedule-specification.html for more info.
# ---
backups_schedule = "@daily-random"

# Cron schedule for prune task (when old backups are discarded).
# See https://docs.k8up.io/k8up/references/schedule-specification.html for more info.
# ---
backups_prune_schedule = "@daily-random"

# Backups retention policy - how many last automatic backups to save.
# Helps to save storage and to get rid of old backups as they age.
# Manually created backups (without autobackup tag) are not discarded.
#
# You can set keepLast, keepHourly, keepDaily, keepWeekly, keepMonthly and keepYearly.
# ---
backups_retention = {
  # How many daily snapshots to save.
  # ---
  keepDaily = 7
}

# Whether to delete on destroy all backup data from bucket or not.
cleanup_bucket_on_destroy = false

# endregion Backups

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                      Kubernetes                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region k8s

# Version of the k8s to be used.
# Set to null or don't set to use Nebius default (recommended), or specify explicitly
# ---
k8s_version = 1.31

# SSH user credentials for accessing k8s nodes.
# That option add public ip address to every node.
# By default, empty list.
# ---
# k8s_cluster_node_ssh_access_users = [{
#   name = "<USER1>"
#   public_keys = [
#     "<ENCRYPTION-METHOD1 HASH1 USER1>",
#     "<ENCRYPTION-METHOD2 HASH2 USER1>",
#   ]
# }]

# endregion k8s
