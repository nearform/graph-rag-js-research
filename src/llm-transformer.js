import { z } from 'zod'
import { ChatPromptTemplate } from '@langchain/core/prompts'
//import { GraphDocument, Node, Relationship } from '@langchain/community/dist/graphs/graph_document';
// Todo: Ideally, graph-document.js should come from the LangChain JS library. But for some reason, it is not available in the library.
import { GraphDocument, Node, Relationship } from './graph-document.js'

const systemPrompt =
  '# Knowledge Graph Instructions for GPT-4\n' +
  '## 1. Overview\n' +
  'You are a top-tier algorithm designed for extracting information in structured ' +
  'formats to build a knowledge graph.\n' +
  'Try to capture as much information from the text as possible without ' +
  'sacrifing accuracy. Do not add any information that is not explicitly ' +
  'mentioned in the text\n' +
  '- **Nodes** represent entities and concepts.\n' +
  '- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n' +
  'accessible for a vast audience.\n' +
  '## 2. Labeling Nodes\n' +
  '- **Consistency**: Ensure you use available types for node labels.\n' +
  'Ensure you use basic or elementary types for node labels.\n' +
  '- For example, when you identify an entity representing a person, ' +
  "always label it as **'person'**. Avoid using more specific terms " +
  "like 'mathematician' or 'scientist'" +
  '  - **Node IDs**: Never utilize integers as node IDs. Node IDs should be ' +
  'names or human-readable identifiers found in the text.\n' +
  '- **Relationships** represent connections between entities or concepts.\n' +
  'Ensure consistency and generality in relationship types when constructing ' +
  'knowledge graphs. Instead of using specific and momentary types ' +
  "such as 'BECAME_PROFESSOR', use more general and timeless relationship types " +
  "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n" +
  '## 3. Coreference Resolution\n' +
  "- **Maintain Entity Consistency**: When extracting entities, it's vital to " +
  'ensure consistency.\n' +
  'If an entity, such as "John Doe", is mentioned multiple times in the text ' +
  'but is referred to by different names or pronouns (e.g., "Joe", "he"),' +
  'always use the most complete identifier for that entity throughout the ' +
  'knowledge graph. In this example, use "John Doe" as the entity ID.\n' +
  'Remember, the knowledge graph should be coherent and easily understandable, ' +
  'so maintaining consistency in entity references is crucial.\n' +
  '## 4. Strict Compliance\n' +
  'Adhere to the rules strictly. Non-compliance will result in termination.'

const defaultPrompt = ChatPromptTemplate.fromMessages([
  ['system', systemPrompt],
  [
    'human',
    'Tip: Make sure to answer in the correct format and do not include any explanations. Use the given format to extract information from the following input: {input}'
  ]
])

// Utility function to conditionally create a field with an enum constraint.
function optionalEnumField(enumValues, description, isRel) {
  if (enumValues && enumValues.length > 0) {
    return z
      .enum(enumValues)
      .describe(`${description}. Available options are ${enumValues}`)
  } else {
    const nodeInfo =
      'Ensure you use basic or elementary types for node labels.\n' +
      'For example, when you identify an entity representing a person, ' +
      "always label it as **'Person'**. Avoid using more specific terms " +
      "like 'Mathematician' or 'Scientist'"
    const relInfo =
      'Instead of using specific and momentary types such as ' +
      "'BECAME_PROFESSOR', use more general and timeless relationship types like " +
      "'PROFESSOR'. However, do not sacrifice any accuracy for generality"
    const additionalInfo = isRel ? relInfo : nodeInfo
    return z.string().describe(description + additionalInfo)
  }
}

function createSimpleModel(nodeLabels = [], relTypes = []) {
  return z
    .object({
      nodes: z
        .array(
          z.object({
            id: z
              .string()
              .describe('Name or human-readable unique identifier.'),
            type: optionalEnumField(
              nodeLabels,
              'The type or label of the node.'
            )
          })
        )
        .describe('List of nodes'),
      relationships: z
        .array(
          z.object({
            sourceNodeId: z
              .string()
              .describe(
                'Name or human-readable unique identifier of source node'
              ),
            sourceNodeType: optionalEnumField(
              nodeLabels,
              'The type or label of the source node.'
            ),
            targetNodeId: z
              .string()
              .describe(
                'Name or human-readable unique identifier of target node'
              ),
            targetNodeType: optionalEnumField(
              nodeLabels,
              'The type or label of the target node.'
            ),
            type: optionalEnumField(
              relTypes,
              'The type of the relationship.',
              true
            )
          })
        )
        .describe('List of relationships')
    })
    .describe('Identifying information about entities.')
}

function toTitleCase(phrase) {
  return phrase
    .toLowerCase()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function capitalize(phrase) {
  return phrase.charAt(0).toUpperCase() + phrase.slice(1)
}

function mapToBaseNode(nodes) {
  return (nodes ?? []).map(node => {
    return new Node({
      id: toTitleCase(node.id),
      type: capitalize(node.type)
    })
  })
}

function mapToBaseRelationship(relationships) {
  return (relationships ?? []).map(rel => {
    return new Relationship({
      source: new Node({
        id: toTitleCase(rel.sourceNodeId),
        type: capitalize(rel.sourceNodeType)
      }),
      target: new Node({
        id: toTitleCase(rel.targetNodeId),
        type: capitalize(rel.targetNodeType)
      }),
      type: rel.type.replace(' ', '_').toUpperCase()
    })
  })
}

export class LLMGraphTransformer {
  constructor(
    llm,
    allowedNodes = [],
    allowedRelationships = [],
    prompt = defaultPrompt,
    strictMode = true
  ) {
    if (!llm.withStructuredOutput) {
      throw new Error(
        "The specified LLM does not support the 'withStructuredOutput'. Please ensure you are using an LLM that supports this feature."
      )
    }
    this.allowedNodes = allowedNodes
    this.allowedRelationships = allowedRelationships
    this.strictMode = strictMode

    const graphSchema = createSimpleModel(allowedNodes, allowedRelationships)
    this.chain = prompt.pipe(llm.withStructuredOutput(graphSchema))
  }

  async processResponse(document) {
    const text = document.pageContent
    const rawSchema = await this.chain.invoke({ input: text })
    let nodes = mapToBaseNode(rawSchema.nodes)
    let relationships = mapToBaseRelationship(rawSchema.relationships)

    // Strict mode filtering
    if (
      this.strictMode &&
      (this.allowedNodes.length > 0 || this.allowedRelationships.length > 0)
    ) {
      if (this.allowedNodes.length > 0) {
        nodes = nodes.filter(node => this.allowedNodes.includes(node.type))
        relationships = relationships.filter(
          rel =>
            this.allowedNodes.includes(rel.source.type) &&
            this.allowedNodes.includes(rel.target.type)
        )
      }
      if (this.allowedRelationships.length > 0) {
        relationships = relationships.filter(rel =>
          this.allowedRelationships.includes(rel.type)
        )
      }
    }

    // Todo: Remove the below line once the issue is fixed in the LangChain JS library.
    // https://github.com/langchain-ai/langchainjs/blob/main/libs/langchain-community/src/graphs/neo4j_graph.ts#L46
    document.page_content = document.pageContent

    return new GraphDocument({ nodes, relationships, source: document })
  }

  async convertToGraphDocuments(documents) {
    const results = []
    for (const document of documents) {
      try {
        const result = await this.processResponse(document)
        console.log(result.source.pageContent, '############')
        results.push(result)
      } catch (e) {
        console.log(e)
      }
    }
    return results
  }
}
