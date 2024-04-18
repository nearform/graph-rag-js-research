import { z } from 'zod'
import 'neo4j-driver'
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph'
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts'
import { ChatOpenAI } from '@langchain/openai'
import { Neo4jVectorStore } from '@langchain/community/vectorstores/neo4j_vector'
import { OpenAIEmbeddings } from '@langchain/openai'
import { createStructuredOutputRunnable } from 'langchain/chains/openai_functions'
import {
  RunnablePassthrough,
  RunnableSequence
} from '@langchain/core/runnables'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { generateFullTextQuery } from './utils.js'

const url = process.env.NEO4J_URI
const username = process.env.NEO4J_USERNAME
const password = process.env.NEO4J_PASSWORD
const openAIApiKey = process.env.OPENAI_API_KEY
const modelName = 'gpt-3.5-turbo-0125'

const llm = new ChatOpenAI({
  openAIApiKey,
  modelName
})

const graph = await Neo4jGraph.initialize({ url, username, password })

const neo4jVectorIndex = await Neo4jVectorStore.fromExistingGraph(
  new OpenAIEmbeddings({ openAIApiKey }),
  {
    url,
    username,
    password,
    searchType: 'hybrid',
    nodeLabel: 'Document',
    textNodeProperties: ['text'],
    embeddingNodeProperty: 'embedding'
  }
)

// Identifying information about entities.
const entitiesSchema = z
  .object({
    names: z
      .array(z.string())
      .describe(
        'All the person, organization, or business entities that appear in the text'
      )
  })
  .describe('Identifying information about entities.')

const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    'You are extracting organization and person entities from the text.'
  ],
  [
    'human',
    'Use the given format to extract information from the following input: {question}'
  ]
])

const entityChain = createStructuredOutputRunnable({
  outputSchema: entitiesSchema,
  prompt,
  llm
})

await graph.query(
  'CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]'
)

/**
 * Collects the neighborhood of entities mentioned in the question.
 *
 * @param {string} question
 * @returns {string}
 */
async function structuredRetriever(question) {
  let result = ''
  const entities = await entityChain.invoke({ question })

  for (const entity of entities.names) {
    const response = await graph.query(
      `CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
      YIELD node,score
      CALL {
        MATCH (node)-[r:!MENTIONS]->(neighbor)
        RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS 
        output
        UNION
        MATCH (node)<-[r:!MENTIONS]-(neighbor)
        RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS 
        output
      }
      RETURN output LIMIT 50`,
      { query: generateFullTextQuery(entity) }
    )

    result += response.map(el => el.output).join('\n') + '\n'
  }

  return result
}

async function retriever(question) {
  console.log('Standalone Question - ' + question)
  const structuredData = await structuredRetriever(question)

  const similaritySearchResults =
    await neo4jVectorIndex.similaritySearch(question)
  const unstructuredData = similaritySearchResults.map(el => el.pageContent)

  const finalData = `Structured data:
  ${structuredData}
  Unstructured data:
  ${unstructuredData.map(content => `#Document ${content}`).join('\n')}
      `
  return finalData
}

const standaloneTemplate = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{conversationHistory}
Follow Up Input: {question}
Standalone question:`

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer any question based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I am sorry, I don't know the answer to that.". And don't try to makeup the answer. Always speak as you are chatting to a friend

context:{context}
question: {question}
answer:`

const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

const standalonePrompt = PromptTemplate.fromTemplate(standaloneTemplate)

const standaloneQuestionChain = standalonePrompt
  .pipe(llm)
  .pipe(new StringOutputParser())

const retrieverChain = RunnableSequence.from([
  prevResult => prevResult.standaloneQuestion,
  retriever
])

const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser())

const chain = RunnableSequence.from([
  {
    standaloneQuestion: standaloneQuestionChain,
    orignalInput: new RunnablePassthrough()
  },
  {
    context: retrieverChain,
    question: ({ orignalInput }) => orignalInput.question,
    conversationHistory: ({ orignalInput }) => orignalInput.conversationHistory
  },
  answerChain
])

function formatChatHistory(chatHistory) {
  return chatHistory
    .map((message, i) => {
      if (i % 2 === 0) {
        return `Human: ${message}`
      }
      return `AI: ${message}`
    })
    .join('\n')
}

const conversationHistory = []

function logResult(result) {
  console.log(`Search Result - ${result}\n`)
}

async function ask(question) {
  console.log(`Search Query - ${question}`)
  const answer = await chain.invoke({
    question,
    conversationHistory: formatChatHistory(conversationHistory)
  })
  conversationHistory.push(question)
  conversationHistory.push(answer)
  logResult(answer)
}

await ask('Loki is a native of which planet?')
await ask('What is the name of his brother?')
await ask('Who is the villain among the two?')
await ask('Who is Tony Stark?')
await ask('Does he own the Stark Tower?')

await graph.close()
await neo4jVectorIndex.close()
