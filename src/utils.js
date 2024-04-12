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
export function generateFullTextQuery(input) {
  let fullTextQuery = ''

  const words = removeLuceneChars(input).split(' ')
  for (let i = 0; i < words.length - 1; i++) {
    fullTextQuery += ` ${words[i]}~2 AND`
  }
  fullTextQuery += ` ${words[words.length - 1]}~2`

  return fullTextQuery.trim()
}
