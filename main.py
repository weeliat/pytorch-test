from clearml import Task
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
import datetime


def run_training(rank, world_size):
    print(f"Running training on rank {rank} out of {world_size} processes")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for this training")

    # Set the device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Set up distributed process group
    print(f"Initializing process group: rank={rank}, world_size={world_size}")

    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize process group with timeout
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=60),  # 60 second timeout
        )
        print(f"Rank {rank}: Successfully initialized process group")
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group: {e}")
        raise

    # Load data with proper distribution
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Use DistributedSampler to partition the dataset
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=sampler)

    # Create model, loss function, and optimizer
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    ).to(
        device
    )  # Move to GPU immediately

    # Use DistributedDataParallel with device_ids explicitly set
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Add this at the start of run_training
    print(f"Rank {rank}: CUDA available: {torch.cuda.is_available()}")
    print(f"Rank {rank}: Current device: {torch.cuda.current_device()}")
    print(f"Rank {rank}: Device count: {torch.cuda.device_count()}")
    print(f"Rank {rank}: Device name: {torch.cuda.get_device_name(rank)}")

    # Training loop
    for epoch in range(5):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    f"Rank {rank}, Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}"
                )
                # Log metrics to ClearML
                Task.current_task().get_logger().report_scalar(
                    "loss",
                    "train",
                    value=running_loss / 100,
                    iteration=epoch * len(dataloader) + i,
                )
                running_loss = 0.0

        # Upload artifacts from each worker
        Task.current_task().upload_artifact(
            name=f"model_state_rank_{rank}_epoch_{epoch}",
            artifact_object=model.state_dict(),
        )

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    # Initialize the ClearML task
    task = Task.init(project_name="Tests", task_name="pytorch_distributed_training")

    # Execute remotely
    task.execute_remotely(queue_name="default")

    # Launch multi-node execution
    config = task.launch_multi_node(total_num_nodes=2, port=29500)

    # Get rank and world size from config
    rank = config.get("node_rank")
    world_size = config.get("total_num_nodes")

    print(f"Starting process with rank {rank} out of {world_size} processes")

    # Run the training function
    run_training(rank, world_size)
