import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

async def main():
    # Connect to the Temporal server
    client = await Client.connect("localhost:7233")  # await the connection

    # Create the workflow client
    workflow_client = client.new_workflow_stub(AudioClassificationWorkflow)

    # Define the input data (audio file for processing)
    input_data = {"file_name": "path_to_your_audio_file.wav"}

    # Start the workflow execution
    result = await workflow_client.run(input_data)  # await the workflow execution

    # Output the result
    print(f"Workflow result: {result}")

# Run the main function using asyncio
asyncio.run(main())
