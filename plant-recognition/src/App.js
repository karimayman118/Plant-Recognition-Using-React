import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [image, setImage] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async () => {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data.prediction);
            setImage(response.data.image);
            setError(null);
        } catch (err) {
            setError('Error predicting: ' + err.response?.data?.error || err.message);
            setResult(null);
            setImage(null);
        }
    };

    return (
        <div className="app-container">
            <div className="header">
                <h1>Plant Recognition</h1>
            </div>
            <div className="form-container">
                <input type="file" onChange={handleFileChange} />
                <button onClick={handleSubmit}>Upload and Predict</button>
            </div>
            {result && (
                <div className="result-container">
                    <h2>Predicted Plant Type: {result}</h2>
                    {image && (
                        <div className="image-container">
                            <img src={`data:image/jpeg;base64,${image}`} alt="Uploaded Plant" />
                        </div>
                    )}
                </div>
            )}
            {error && <p className="error">{error}</p>}
        </div>
    );
};

export default App;
