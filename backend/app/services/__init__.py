"""
Business services module
========================

Contains all domain-specific service classes that implement the core
MiroFish intelligence pipeline:

Knowledge graph
---------------
- :class:`OntologyGenerator`       — LLM-driven entity/relationship extraction
- :class:`GraphBuilderService`     — Builds a Zep knowledge graph from extracted text
- :class:`TextProcessor`           — Chunk and preprocess raw document text

Entity retrieval
----------------
- :class:`ZepEntityReader`         — Read and filter typed entities from a Zep graph
- :class:`ZepToolsService`         — InsightForge / PanoramaSearch / QuickSearch / Interview tools

OASIS simulation
----------------
- :class:`OasisProfileGenerator`   — Generate agent social profiles from Zep entities
- :class:`SimulationManager`       — Create, prepare, and delete simulation runs (disk + cache)
- :class:`SimulationConfigGenerator` — Build OASIS YAML/JSON configuration files
- :class:`SimulationRunner`        — Launch & monitor OASIS simulation sub-processes
- :class:`ZepGraphMemoryManager`   — Persist agent actions back into the Zep graph

Report generation
-----------------
- :mod:`report_agent`              — ReACT-mode multi-section report agent (ReportAgent /
                                     ReportManager); also owns ``render_html`` and the
                                     ``_inline`` Markdown converter.

IPC
---
- :class:`SimulationIPCClient`     — Send commands to a running simulation process
- :class:`SimulationIPCServer`     — Receive commands inside the simulation process
"""

from .ontology_generator import OntologyGenerator
from .graph_builder import GraphBuilderService
from .text_processor import TextProcessor
from .zep_entity_reader import ZepEntityReader, EntityNode, FilteredEntities
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_manager import SimulationManager, SimulationState, SimulationStatus
from .simulation_config_generator import (
    SimulationConfigGenerator, 
    SimulationParameters,
    AgentActivityConfig,
    TimeSimulationConfig,
    EventConfig,
    PlatformConfig
)
from .simulation_runner import (
    SimulationRunner,
    SimulationRunState,
    RunnerStatus,
    AgentAction,
    RoundSummary
)
from .zep_graph_memory_updater import (
    ZepGraphMemoryUpdater,
    ZepGraphMemoryManager,
    AgentActivity
)
from .simulation_ipc import (
    SimulationIPCClient,
    SimulationIPCServer,
    IPCCommand,
    IPCResponse,
    CommandType,
    CommandStatus
)

__all__ = [
    'OntologyGenerator', 
    'GraphBuilderService', 
    'TextProcessor',
    'ZepEntityReader',
    'EntityNode',
    'FilteredEntities',
    'OasisProfileGenerator',
    'OasisAgentProfile',
    'SimulationManager',
    'SimulationState',
    'SimulationStatus',
    'SimulationConfigGenerator',
    'SimulationParameters',
    'AgentActivityConfig',
    'TimeSimulationConfig',
    'EventConfig',
    'PlatformConfig',
    'SimulationRunner',
    'SimulationRunState',
    'RunnerStatus',
    'AgentAction',
    'RoundSummary',
    'ZepGraphMemoryUpdater',
    'ZepGraphMemoryManager',
    'AgentActivity',
    'SimulationIPCClient',
    'SimulationIPCServer',
    'IPCCommand',
    'IPCResponse',
    'CommandType',
    'CommandStatus',
]
