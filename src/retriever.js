import { z } from 'zod'
import 'neo4j-driver'
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { ChatOpenAI } from '@langchain/openai'
import { Neo4jVectorStore } from '@langchain/community/vectorstores/neo4j_vector'
import { OpenAIEmbeddings } from '@langchain/openai'
import { createStructuredOutputRunnable } from 'langchain/chains/openai_functions'
import {
  RunnablePassthrough,
  RunnableSequence
} from '@langchain/core/runnables'
import { StringOutputParser } from '@langchain/core/output_parsers'

const url = process.env.NEO4J_URI
const username = process.env.NEO4J_USERNAME
const password = process.env.NEO4J_PASSWORD
const openAIApiKey = process.env.OPENAI_API_KEY

const graph = await Neo4jGraph.initialize({ url, username, password })

const llm = new ChatOpenAI({
  temperature: 0,
  modelName: 'gpt-3.5-turbo-0125',
  openAIApiKey
})

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

function removeLuceneChars(text) {
  if (text === undefined || text === null) {
    return null
  }
  // Remove Lucene special characters
  const specialChars = [
    '+',
    '-',
    '&',
    '|',
    '!',
    '(',
    ')',
    '{',
    '}',
    '[',
    ']',
    '^',
    '"',
    '~',
    '*',
    '?',
    ':',
    '\\'
  ]
  let modifiedText = text
  for (const char of specialChars) {
    modifiedText = modifiedText.split(char).join(' ')
  }
  return modifiedText.trim()
}

/**
 * Generate a full-text search query for a given input string.
 *
 * This function constructs a query string suitable for a full-text
 * search. It processes the input string by splitting it into words and
 * appending a similarity threshold (~2 changed characters) to each
 * word, then combines them using the AND operator. Useful for mapping
 * entities from user questions to database values, and allows for some
 * misspelings.
 *
 * @param {string} input
 * @returns {string}
 */
function generateFullTextQuery(input) {
  let fullTextQuery = ''

  const words = removeLuceneChars(input).split(' ')
  for (let i = 0; i < words.length - 1; i++) {
    fullTextQuery += ` ${words[i]}~2 AND`
  }
  fullTextQuery += ` ${words[words.length - 1]}~2`

  return fullTextQuery.trim()
}

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
      `CALL db.index.fulltext.queryNodes('entity', $query, 
            {limit:2})
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
  console.log(`Search Query - ${question}`)
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

const template = `Answer the question based only on the following context:
{context}

Question: {question}
`
const promptFromTemplate = ChatPromptTemplate.fromTemplate(template)

const chain = RunnableSequence.from([
  {
    context: retriever,
    question: new RunnablePassthrough()
  },
  promptFromTemplate,
  llm,
  new StringOutputParser()
])

function logResult(result) {
  console.log(`Search Result - ${result}\n`)
}

logResult(await chain.invoke('Who is the billionaire in the Avengers group?'))
logResult(await chain.invoke('Loki is a native of which planet?'))
logResult(await chain.invoke('What is the other name of Bruce Banner?'))
logResult(await chain.invoke('Where is the Stark Tower located?'))
logResult(await chain.invoke('Who is Loki?'))
logResult(await chain.invoke('Who is Jarvis?'))

await graph.close()
await neo4jVectorIndex.close()
