import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: 'sk-ant-HpKRhWq3iAZu6eLSNlz3wvgBcVZZf3KYfE3L4C4UTQA',
  baseURL: 'https://treasury-adsl-entitled-bathroom.trycloudflare.com/v1',
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
