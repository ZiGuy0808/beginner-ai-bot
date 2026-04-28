import Anthropic from '@anthropic-ai/sdk';

// Replace YOUR_KEY_HERE with your Anthropic API key
const ANTHROPIC_API_KEY = 'YOUR_KEY_HERE';

const anthropic = new Anthropic({
  apiKey: ANTHROPIC_API_KEY,
});


async function main() {
  const message = await anthropic.messages.create({
    model: 'claude-3-5-sonnet-20240620',
    max_tokens: 1024,
    messages: [{ role: 'user', content: 'Hello!' }],
  });

  console.log(message.content[0].text);
}

main();
