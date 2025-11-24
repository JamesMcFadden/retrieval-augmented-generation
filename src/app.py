from flask import Flask, render_template, request
from app_utils import run_rag_pipeline

app = Flask(__name__, template_folder="../templates")

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    query = ""

    if request.method == "POST":
        query = request.form["query"]
        response = run_rag_pipeline(query)

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)