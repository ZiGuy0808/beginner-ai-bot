import { OpenAI } from 'openai';

const openai = new OpenAI({
  apiKey: 'sk-d5Fw6wuQWm8KIwgP1MfC1NdNMUCARjckJR9D1J600i8',
  baseURL: 'https://valuation-asked-staffing-habits.trycloudflare.com/v1',
});

async function main() {
  const completion = await openai.chat.completions.create({
    messages: [{ role: 'user', content: 'Hello!' }],
    model: 'claude-3-5-sonnet-20240620',
  });

  console.log(completion.choices[0].message.content);
}

main();
