"""Server-side dynamic workflow machinery.

Provides the spec-validation, subgraph-validation, and graph-assembly logic for
dynamic workflows: validating a goal + driver spec, structurally validating the
planner's emitted subgraph (including reference-library resolution), and
assembling each round's native graph (emitted subgraph plus observation and
proposer ops).
"""
