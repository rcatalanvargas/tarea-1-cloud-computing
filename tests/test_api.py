from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_STUDENT = {
    "Age": 22,
    "Gender": "Female",
    "University_Year": "Senior",
    "Major": "Data Science",
    "Attendance_Percentage": 85,
    "Study_Hours_Per_Week": 25,
    "CGPA": 3.4,
    "Programming_Skill": 8,
    "Projects_Completed": 10,
    "Certifications": 3,
    "Hackathons": 4,
    "GitHub_Profile": "Yes",
    "Internships": 4,
    "Leadership_Experience": "Yes",
    "LinkedIn_Profile": "Yes",
    "Resume_Score": 95,
    "Communication_Skills": 8,
    "Teamwork": 8,
    "Problem_Solving": 9,
    "English_Proficiency": "Advanced",
    "Interview_Score": 88,
}


LOW_PROFILE_STUDENT = {
    "Age": 20,
    "Gender": "Male",
    "University_Year": "Freshman",
    "Major": "Information Technology",
    "Attendance_Percentage": 55,
    "Study_Hours_Per_Week": 6,
    "CGPA": 2.1,
    "Programming_Skill": 1,
    "Projects_Completed": 0,
    "Certifications": 0,
    "Hackathons": 0,
    "GitHub_Profile": "No",
    "Internships": 0,
    "Leadership_Experience": "No",
    "LinkedIn_Profile": "No",
    "Resume_Score": 44,
    "Communication_Skills": 3,
    "Teamwork": 2,
    "Problem_Solving": 1,
    "English_Proficiency": "Basic",
    "Interview_Score": 17,
}


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["loader"] is True


def test_model_info() -> None:
    response = client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["estimator_name"] == "LogisticRegression"
    assert body["target"] == "Placement_Status"
    assert body["input_feature_count"] == 21
    assert body["classes"] == [
        "Not Placed",
        "Placed",
    ]


def test_predict_valid_student() -> None:
    response = client.post(
        "/predict",
        json=VALID_STUDENT,
    )
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] == "Placed"
    assert 0.0 <= body["probability_placed"] <= 1.0

    probabilities = body["class_probabilities"]

    assert set(probabilities) == {
        "Not Placed",
        "Placed",
    }
    assert abs(
        sum(probabilities.values()) - 1.0
    ) < 0.000001


def test_predict_rejects_invalid_age() -> None:
    invalid_student = {
        **VALID_STUDENT,
        "Age": 17,
    }

    response = client.post(
        "/predict",
        json=invalid_student,
    )

    assert response.status_code == 422

    error_details = response.json()["detail"]

    assert any(
        error["loc"][-1] == "Age"
        for error in error_details
    )


def test_predict_batch() -> None:
    response = client.post(
        "/predict-batch",
        json={
            "students": [
                VALID_STUDENT,
                LOW_PROFILE_STUDENT,
            ]
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["prediction_count"] == 2
    assert len(body["predictions"]) == 2

    assert (
        body["predictions"][0]["prediction"]
        == "Placed"
    )
    assert (
        body["predictions"][1]["prediction"]
        == "Not Placed"
    )