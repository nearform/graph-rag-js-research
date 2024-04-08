import 'neo4j-driver'
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph'
import { IMSDBLoader } from 'langchain/document_loaders/web/imsdb'
import { TokenTextSplitter } from 'langchain/text_splitter'
import { ChatOpenAI } from '@langchain/openai'
import { LLMGraphTransformer } from './llm-transformer.js'
import { Document } from '@langchain/core/documents'

const url = process.env.NEO4J_URI
const username = process.env.NEO4J_USERNAME
const password = process.env.NEO4J_PASSWORD
const openAIApiKey = process.env.OPENAI_API_KEY

const graph = await Neo4jGraph.initialize({ url, username, password })

const loader = new IMSDBLoader('https://imsdb.com/scripts/Avengers,-The-(2012).html');
const rawDocs = await loader.load()

// Define chunking strategy
const textSplitter = new TokenTextSplitter({
  chunkSize: 512,
  chunkOverlap: 24
})

let documents = []
for (let i = 0; i < rawDocs.length; i++) {
  const chunks = await textSplitter.splitText(rawDocs[i].pageContent)
  const processedDocs = chunks.map(
    (chunk, index) =>
      new Document({
        pageContent: chunk,
        metadata: {
          a: index + 1,
          ...rawDocs[i].metadata
        }
      })
  )
  documents.push(...processedDocs)
}

const llm = new ChatOpenAI({
  temperature: 0,
  modelName: 'gpt-3.5-turbo-0125',
  openAIApiKey
})

const llmTransformer = new LLMGraphTransformer(llm)
const graphDocuments = await llmTransformer.convertToGraphDocuments(documents)

await graph.addGraphDocuments(graphDocuments, {
  baseEntityLabel: true,
  includeSource: true
})

console.log('Completed adding graph documents!!!')

await graph.close()
