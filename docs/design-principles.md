# Design Principles

SilverLingua follows a biological-inspired architecture that organizes components into distinct layers of complexity and responsibility. This design emerged from the need for a lightweight, type-safe, and intuitive framework for building AI agents without unnecessary complexity or boilerplate.

## Core Philosophy

SilverLingua was created in response to the challenges of working with existing frameworks like LangChain, which often introduce unnecessary complexity and tight coupling. Our core principles are:

- **Lightweight**: Every component serves a clear purpose with minimal overhead
- **Type-safe**: Leveraging Pydantic for runtime type validation and data modeling, ensuring reliable and maintainable code with excellent IDE support
- **Easily extensible**: Universal interfaces make it simple to add or modify functionality
- **Intuitive**: Clear action flows and lifecycle methods that are easy to understand
- **Surgical modification**: Ability to modify specific behaviors without affecting others
- **Minimal boilerplate**: Focus on productive code, not repetitive setup

## Core Architectural Layers

### Atoms

Atoms are the fundamental, indivisible building blocks of SilverLingua. They represent single-purpose, foundational components that can't be broken down further while remaining useful.

**Characteristics:**

- Implement a single, focused piece of functionality
- Zero external dependencies
- Are stateless or manage minimal state
- Completely self-contained
- Serve as pure building blocks for more complex components

**Examples:**

- `Memory`: Basic unit of storage
- `Prompt`: Single unit of instruction or query
- `Role`: Individual behavior definition
- `Tool`: Single capability implementation
- `Tokenizer`: Basic text processing unit

### Molecules

Molecules represent self-contained conceptual units that combine atoms to create more complex but focused components. They focus on "what something is" rather than "what it does."

**Characteristics:**

- Represent individual concepts or relationships
- Manage state and properties of a single conceptual unit
- Can compose with other molecules
- Don't orchestrate complex behaviors
- Focus on representation over action

**Examples:**

- `Notion`: Represents a single thought or concept
- `Link`: Represents a relationship between components (extends Notion)

### Organisms

Organisms are complex components that coordinate multiple molecules and atoms to perform sophisticated tasks. They focus on "what something does" and how it orchestrates behavior.

**Characteristics:**

- Coordinate multiple components
- Implement complex workflows
- Manage system-wide behaviors
- Orchestrate interactions between molecules
- Focus on action and coordination

**Examples:**

- `Idearium`: Manages a network of thoughts and their relationships

### Templates

Templates define core interfaces and abstract base classes that ensure consistency across the system. They establish patterns that other components must follow while keeping implementation details flexible.

**Characteristics:**

- Define standard interfaces with intuitive lifecycle methods
- Provide minimal base implementations
- Ensure architectural consistency
- Enable modularity and extensibility
- Allow surgical modifications of specific behaviors

**Examples:**

- `Agent`: Base template for AI agents
- `Model`: Interface for language model implementations, with clear lifecycle methods

## Design Guidelines

### When to Create Each Type

#### Create an Atom when:

- You need a fundamental, single-purpose component
- The functionality can't be meaningfully broken down further
- It will be reused across many different contexts
- It can exist without any external dependencies

#### Create a Molecule when:

- You need to represent a concept or relationship
- You're combining multiple atoms into a cohesive unit
- The focus is on "what something is"
- The component needs to maintain its own state

#### Create an Organism when:

- You need to coordinate multiple molecules or atoms
- You're implementing complex workflows
- The focus is on "what something does"
- You need to manage system-wide behavior

#### Create a Template when:

- You're defining a core interface that others will implement
- You need to ensure consistency across multiple implementations
- You're establishing a fundamental pattern with clear lifecycle methods

### Composition Rules

1. Atoms must be completely self-contained with zero external dependencies
2. Molecules can depend on atoms and compose with other molecules
3. Organisms can use any components but should focus on coordination
4. Templates should minimize dependencies and focus on clear, intuitive interfaces

### Naming Conventions

- Atoms: Simple, foundational concepts (`Memory`, `Tool`, `Role`)
- Molecules: Conceptual nouns (`Notion`, `Link`)
- Organisms: Action-oriented or system concepts (`Idearium`)
- Templates: Abstract patterns (`Agent`, `Model`)

## Contributing

When contributing new components to SilverLingua:

1. Identify the appropriate layer for your component based on its purpose and complexity
2. Follow the design guidelines for that layer
3. Maintain clear separation of concerns
4. Keep it simple - if it feels complicated, it probably needs to be broken down
5. Consider composition over inheritance when possible
6. Ensure modifications can be made surgically without side effects

## Real-World Analogies

Think of SilverLingua's architecture like a biological system:

- Atoms are like proteins: fundamental, self-contained building blocks
- Molecules are like cells: self-contained units with specific purposes
- Organisms are like organs: coordinating multiple parts for complex functions
- Templates are like DNA: blueprints that define how components should be structured

This biological metaphor helps maintain consistency in design decisions and makes the system more intuitive to understand and extend.
