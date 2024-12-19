# import asyncio
#
# from autogen_core import RoutedAgent, rpc, MessageContext
# from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
#
# from agents.common import OverviewInfo, WorkerInput
# from utils.data_utils import analyze_full_data
# import pandas as pd
#
# from utils.settings import settings
#
# worker_runtime_client = GrpcWorkerAgentRuntime(host_address=settings.worker_host_address)
#
#
# class WorkerAgent(RoutedAgent):
#     def __init__(self, description: str = "Worker Agent"):
#         super().__init__(description=description)
#
#     @rpc
#     async def perform_analysis(self, message: WorkerInput, ctx: MessageContext) -> OverviewInfo:
#         dataframe = pd.DataFrame.from_dict(message.dataframe)
#         # Perform the analysis
#         return analyze_full_data(dataframe)
#
#
# async def main():
#     worker_runtime_client.start()
#
#     # host_address = "localhost:50051"
#     # host = GrpcWorkerAgentRuntimeHost(address=host_address)
#     # host.start()
#
#     await worker_runtime_client.register_factory(
#         type=WorkerAgent("worker-agent"), agent_factory=lambda: WorkerAgent(), expected_class=WorkerAgent
#     )
#     await worker_runtime_client.stop_when_signal()
#
#
# if __name__ == "__main__":
#     asyncio.run(main())
