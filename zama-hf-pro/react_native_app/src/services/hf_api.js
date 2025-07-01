import axios from 'axios';

const HF_API_ENDPOINT = 'https://api-inference.huggingface.co/models/';
const ZAMAI_API_ENDPOINT = 'https://your-api.com/api/';

export const queryModel = async (modelId, inputs, token) => {
  try {
    const response = await axios.post(
      `${HF_API_ENDPOINT}${modelId}`,
      { inputs },
      { 
        headers: { 
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        } 
      }
    );
    return response.data;
  } catch (error) {
    console.error('HF API Error:', error);
    throw error;
  }
};

export const generateText = async (prompt, model = 'tasal9/ZamAI-Mistral-7B-Pashto') => {
  try {
    const response = await axios.post(
      `${ZAMAI_API_ENDPOINT}generate`,
      { prompt, model },
      {
        headers: {
          Authorization: `Bearer ${process.env.REACT_APP_HF_TOKEN}`,
          'Content-Type': 'application/json'
        }
      }
    );
    return response.data.response;
  } catch (error) {
    console.error('ZamAI API Error:', error);
    throw error;
  }
};
