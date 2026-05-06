import './App.css'
import DocumentUpload from './components/DocumentUpload'
import QueryPanel from './components/QueryPanel'

function App() {
  return (
    <div className="rag-app">
      <header className="rag-header">
        <h1 className="!mt-0">RAG workspace</h1>
        <p className="rag-lead">
          Store documents in the vector DB, then ask questions grounded on retrieved data.
        </p>
      </header>
      <main className="rag-main">
        <DocumentUpload />
        <QueryPanel />
      </main>
    </div>
  )
}

export default App
