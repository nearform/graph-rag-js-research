import { z } from 'zod';
import 'neo4j-driver';
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';
import { Neo4jVectorStore } from '@langchain/community/vectorstores/neo4j_vector';
import { OpenAIEmbeddings } from '@langchain/openai';
import { createStructuredOutputRunnable } from 'langchain/chains/openai_functions';
import { RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { AIMessage, HumanMessage } from '@langchain/core/messages';


import { IMSDBLoader } from 'langchain/document_loaders/web/imsdb';
import { TokenTextSplitter } from 'langchain/text_splitter';
import { Document } from "@langchain/core/documents";
import { LLMGraphTransformer } from './llm-transformer.js';


const url = process.env.NEO4J_URI;
const username = process.env.NEO4J_USERNAME;
const password = process.env.NEO4J_PASSWORD;
const openAIApiKey = process.env.OPENAI_API_KEY;

const graph = await Neo4jGraph.initialize({ url, username, password });

const llm = new ChatOpenAI({
    temperature: 0,
    modelName: 'gpt-3.5-turbo-0125',
    openAIApiKey
});

/* Loader code start */
const loader = new IMSDBLoader('https://imsdb.com/scripts/Things-My-Father-Never-Taught-Me,-The.html');
const rawDocs = await loader.load();

const textSplitter = new TokenTextSplitter({
    chunkSize: 512,
    chunkOverlap: 24
});

let documents = [];
for (let i = 0; i < rawDocs.length; i++) {
    const chunks = await textSplitter.splitText(rawDocs[i].pageContent);
    const processedDocs = chunks.map((chunk) => (new Document({
        pageContent: chunk,
        metadata: rawDocs[i].metadata
    })));
    documents.push(...processedDocs);
}

const llmTransformer = new LLMGraphTransformer(llm)
const graphDocuments = await llmTransformer.convertToGraphDocuments(documents);

console.log(graphDocuments.length, '.......graphDocuments');

await graph.addGraphDocuments(graphDocuments, {
    baseEntityLabel: true,
    includeSource: true
});

console.log('Completed adding graph documents!!!');
/* Loader code end */

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
);

console.log('Completed adding unstructured documents !!!');

// Identifying information about entities.
const entitiesSchema = z.object({
    names: z.array(z.string()).describe('All the person, organization, or business entities that appear in the text')
}).describe('Identifying information about entities.');

const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are extracting organization and person entities from the text.'],
    ['human', 'Use the given format to extract information from the following input: {question}']
]);

const entityChain = createStructuredOutputRunnable({
    outputSchema: entitiesSchema,
    prompt,
    llm
});

await graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


function removeLuceneChars(text) {
    if (text === undefined || text === null) {
        return null;
    }
    // Remove Lucene special characters
    const specialChars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
    ];
    let modifiedText = text;
    for (const char of specialChars) {
        modifiedText = modifiedText.split(char).join(" ");
    }
    return modifiedText.trim();
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
    let fullTextQuery = '';

    const words = removeLuceneChars(input).split(' ');
    for (let i = 0; i < words.length - 1; i++) {
        fullTextQuery += ` ${words[i]}~2 AND`;
    }
    fullTextQuery += ` ${words[words.length - 1]}~2`;

    return fullTextQuery.trim();
}

/**
 * Collects the neighborhood of entities mentioned in the question.
 * 
 * @param {string} question 
 * @returns {string}
 */
async function structuredRetriever(question) {
    let result = '';
    const entities = await entityChain.invoke({ question });
    //console.log(entities, '............entities');

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
            {"query": generateFullTextQuery(entity)}
        );

        //console.log(generateFullTextQuery(entity), '.............generateFullTextQuery........', entity);

        result += response.map((el) => el.output).join('\n') + '\n';
    }

    return result;
}

async function retriever(question) {
    console.log(`Search Query - ${question}`);
    const structuredData = await structuredRetriever(question);

    const similaritySearchResults = await neo4jVectorIndex.similaritySearch(question);
    //console.log(similaritySearchResults, '.................similaritySearchResults');
    const unstructuredData = similaritySearchResults.map((el) => el.pageContent);

    const finalData = `Structured data:
${structuredData}
Unstructured data:
${unstructuredData.map(content => `#Document ${content}`).join("\n")}
    `;

    console.log(finalData, '.............finalData');
    return finalData;
}

/*const _template = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;
const CONDENSE_QUESTION_PROMPT = PromptTemplate.fromTemplate(_template);

function formatChatHistory(chatHistory) {
    const buffer = [];
    for (const [human, ai] of chatHistory) {
        buffer.push(new HumanMessage({ content: human }));
        buffer.push(new AIMessage({ content: ai }));
    }
    return buffer;
}

const _search_query = RunnableBranch.from([
    // If input includes chat_history, we condense it with the follow-up question
    [
        RunnableLambda.from((x) => Boolean(x.chat_history)).withConfig({
            runName: "HasChatHistoryCheck",
        }), // Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign({
            chat_history: (x) => formatChatHistory(x.chat_history),
        })
        .pipe(CONDENSE_QUESTION_PROMPT)
        .pipe(llm)
        .pipe(new StringOutputParser())
    ],
    // Else, we have no chat history, so just pass through the question
    RunnableLambda.from((x) => x.question)
]);

const template = `Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:`;
const promptFromTemplate = ChatPromptTemplate.fromTemplate(template);

const chain = RunnableSequence.from([
    {
        context: _search_query.pipe(retriever),
        question: new RunnablePassthrough(),
    },
    promptFromTemplate,
    llm,
    new StringOutputParser()
]);*/

// Non Chaining working code
const template = `Answer the question based only on the following context:
{context}

Question: {question}
`;
const promptFromTemplate = ChatPromptTemplate.fromTemplate(template);

const chain = RunnableSequence.from([
    {
        context: retriever,
        question: new RunnablePassthrough(),
    },
    promptFromTemplate,
    llm,
    new StringOutputParser(),
]);

function logResult(result) {
    console.log(`Search Result - ${result}\n`);
}

logResult(await chain.invoke('Who is the father of Melvin?'));
logResult(await chain.invoke('Who is Mary?'));
logResult(await chain.invoke('What happens in the first meeting between Mary and Melvin?'));

await graph.close();
await neo4jVectorIndex.close();
