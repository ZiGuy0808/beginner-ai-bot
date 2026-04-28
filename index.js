import Anthropic from '@anthropic-ai/sdk';

const ANTTHROPIC_API_KEY = 'sk-ant-HpKRhWq3iAZu6eLSNlz3wvgBcVZZf3KYfE3L4C4UTQA';

const anthropic = new Anthropic({
  apiKey: ANTTHROPIC_API_KEY,
  baseURL: 'https://valuation-asked-staffing-habits.trycloudflare.com/v1',
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
