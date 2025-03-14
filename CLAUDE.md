# CLAUDE.md - Project Guidelines

## Project Overview
Experimental Node-based Neural Network - A novel architecture designed to create dynamic concept nodes that form meaning-based associations, starting with text-only implementation before expanding to multi-modal inputs.

## Development Commands
- Currently no established build/test commands (project in conceptual phase)
- When implemented, consider: `npm run dev` for development server

## Code Style Guidelines
- Prefer functional programming paradigms for data transformation pipelines
- Use descriptive variable names reflecting concepts being modeled
- Implement type safety (TypeScript or Python type hints)
- Document node structures, propagation mechanisms, and conceptual mappings
- Error handling should gracefully degrade while preserving network state
- For imports, group by: (1) standard libraries, (2) external dependencies, (3) internal modules
- Aim for modular architecture allowing future extension to multi-modal inputs

## Architecture Principles
- Implement capsule network concepts for preserving hierarchical relationships
- Consider elastic weight consolidation for continuous learning without catastrophic forgetting
- Use graph-based message passing for node propagation mechanisms
- Balance explicit concept nodes with distributed neural representations