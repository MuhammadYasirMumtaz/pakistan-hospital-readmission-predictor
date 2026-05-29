from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymysql
from sqlalchemy import create_engine, text
import os
import base64
from io import BytesIO
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# LOAD MODEL AND FEATURES
model = joblib.load("models/readmission_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
threshold = joblib.load("models/best_threshold.pkl")

# DATABASE CONNECTION
DB_PASSWORD = "admin123789"
engine = create_engine(f"mysql+pymysql://root:{DB_PASSWORD}@localhost:3306/hospital_db")

print("Model loaded successfully")
print(f"Features: {len(feature_names)}")
print(f"Threshold: {threshold}")

# HOME ROUTE
@app.route('/')
def home():
    return render_template("index.html")

# PREDICT ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'race': request.form.get('race', 'Caucasian'),
            'gender': request.form.get('gender', 'Male'),
            'age': request.form.get('age', '[50-60)'),
            'weight': request.form.get('weight', 'Unknown'),
            'admission_type_id': int(request.form.get('admission_type_id', 1)),
            'discharge_disposition_id': int(request.form.get('discharge_disposition_id', 1)),
            'admission_source_id': int(request.form.get('admission_source_id', 7)),
            'time_in_hospital': int(request.form.get('time_in_hospital', 3)),
            'payer_code': request.form.get('payer_code', 'Unknown'),
            'medical_specialty': request.form.get('medical_specialty', 'Unknown'),
            'num_lab_procedures': int(request.form.get('num_lab_procedures', 40)),
            'num_procedures': int(request.form.get('num_procedures', 1)),
            'num_medications': int(request.form.get('num_medications', 10)),
            'number_outpatient': int(request.form.get('number_outpatient', 0)),
            'number_emergency': int(request.form.get('number_emergency', 0)),
            'number_inpatient': int(request.form.get('number_inpatient', 0)),
            'number_diagnoses': int(request.form.get('number_diagnoses', 5)),
            'max_glu_serum': request.form.get('max_glu_serum', 'None'),
            'A1Cresult': request.form.get('A1Cresult', 'None'),
            'metformin': request.form.get('metformin', 'No'),
            'repaglinide': request.form.get('repaglinide', 'No'),
            'nateglinide': request.form.get('nateglinide', 'No'),
            'chlorpropamide': request.form.get('chlorpropamide', 'No'),
            'glimepiride': request.form.get('glimepiride', 'No'),
            'acetohexamide': request.form.get('acetohexamide', 'No'),
            'glipizide': request.form.get('glipizide', 'No'),
            'glyburide': request.form.get('glyburide', 'No'),
            'tolbutamide': request.form.get('tolbutamide', 'No'),
            'pioglitazone': request.form.get('pioglitazone', 'No'),
            'rosiglitazone': request.form.get('rosiglitazone', 'No'),
            'acarbose': request.form.get('acarbose', 'No'),
            'miglitol': request.form.get('miglitol', 'No'),
            'troglitazone': request.form.get('troglitazone', 'No'),
            'tolazamide': request.form.get('tolazamide', 'No'),
            'examide': request.form.get('examide', 'No'),
            'citoglipton': request.form.get('citoglipton', 'No'),
            'insulin': request.form.get('insulin', 'No'),
            'glyburide-metformin': request.form.get('glyburide-metformin', 'No'),
            'glipizide-metformin': request.form.get('glipizide-metformin', 'No'),
            'glimepiride-pioglitazone': request.form.get('glimepiride-pioglitazone', 'No'),
            'metformin-rosiglitazone': request.form.get('metformin-rosiglitazone', 'No'),
            'metformin-pioglitazone': request.form.get('metformin-pioglitazone', 'No'),
            'change': request.form.get('change', 'No'),
            'diabetesMed': request.form.get('diabetesMed', 'No'),
            'diag_1_group': request.form.get('diag_1_group', 'Other'),
            'diag_2_group': request.form.get('diag_2_group', 'Other'),
            'diag_3_group': request.form.get('diag_3_group', 'Other'),
        }

        # Feature engineering
        input_data["total_visits"] = (
            input_data['number_outpatient'] +
            input_data['number_emergency'] +
            input_data['number_inpatient']
        )

        input_data['emergency_ratio'] = (
            input_data['number_emergency'] /
            (input_data['total_visits'] + 1)
        )

        input_data['inpatient_ratio'] = (
            input_data['number_inpatient'] /
            (input_data['total_visits'] + 1)
        )

        input_data['medication_per_visit'] = (
            input_data['num_medications'] /
            (input_data['total_visits'] + 1)
        )

        input_data['lab_procedure_ratio'] = (
            input_data['num_lab_procedures'] /
            (input_data['time_in_hospital'] + 1)
        )

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure correct column order
        input_df = input_df[feature_names]

        # Convert categorical to string
        cat_cols = input_df.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            input_df[col] = input_df[col].astype(str)

        # Make prediction
        risk_probability = model.predict_proba(input_df)[0][1]
        risk_percentage = round(risk_probability * 100, 1)
        prediction = 1 if risk_probability >= threshold else 0

        # Risk level
        if risk_probability >= 0.6:
            risk_level = "High Risk"
            risk_color = "danger"
            recommendation = "Keep patient admitted for further monitoring and treatment."

        elif risk_probability >= threshold:
            risk_level = "Medium Risk"
            risk_color = "warning"
            recommendation = "Monitor closely. Consider extended observation before discharge."

        else:
            risk_level = "Low Risk"
            risk_color = "success"
            recommendation = "Patient can be safely discharged with follow up appointment."

        # Generate SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # SHAP waterfall plot
        plt.figure(figsize=(10, 6))

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )

        plt.title("Why This Patient Was Flagged", fontweight="bold")
        plt.tight_layout()

        # Save to base64 for HTML display
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)

        shap_plot = base64.b64encode(buffer.getvalue()).decode()

        plt.close()

        # Save prediction to MySQL
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO predictions
                    (
                        time_in_hospital,
                        num_medications,
                        number_diagnoses,
                        number_inpatient,
                        num_lab_procedures,
                        num_procedures,
                        number_outpatient,
                        number_emergency,
                        risk_score,
                        risk_level,
                        prediction_result
                    )
                    VALUES (
                        :time_in_hospital,
                        :num_medications,
                        :number_diagnoses,
                        :number_inpatient,
                        :num_lab_procedures,
                        :num_procedures,
                        :number_outpatient,
                        :number_emergency,
                        :risk_score,
                        :risk_level,
                        :prediction_result
                    )
                """), {
                    'time_in_hospital': input_data['time_in_hospital'],
                    'num_medications': input_data['num_medications'],
                    'number_diagnoses': input_data['number_diagnoses'],
                    'number_inpatient': input_data['number_inpatient'],
                    'num_lab_procedures': input_data['num_lab_procedures'],
                    'num_procedures': input_data['num_procedures'],
                    'number_outpatient': input_data['number_outpatient'],
                    'number_emergency': input_data['number_emergency'],
                    'risk_score': float(risk_probability),
                    'risk_level': risk_level,
                    'prediction_result': prediction
                })

                conn.commit()

        except Exception as db_error:
            print(f"DB Error: {db_error}")

        return render_template(
            "result.html",
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            risk_color=risk_color,
            recommendation=recommendation,
            shap_plot=shap_plot,
            input_data=input_data
        )

    except Exception as e:
        return f"Error: {str(e)}", 500


# ANALYTICS ROUTE
@app.route('/analytics')
def analytics():
    try:
        with engine.connect() as conn:

            # Total predictions
            total = conn.execute(
                text("SELECT COUNT(*) FROM predictions")
            ).fetchone()[0]

            # High risk count
            high_risk = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE risk_level='High Risk'")
            ).fetchone()[0]

            # Medium risk count
            medium_risk = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE risk_level='Medium Risk'")
            ).fetchone()[0]

            # Low risk count
            low_risk = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE risk_level='Low Risk'")
            ).fetchone()[0]

            # Recent predictions
            recent = conn.execute(text("""
                SELECT
                    prediction_date,
                    risk_level,
                    risk_score,
                    time_in_hospital,
                    num_medications
                FROM predictions
                ORDER BY prediction_date DESC
                LIMIT 10
            """)).fetchall()

            return render_template(
                'analytics.html',
                total=total,
                high_risk=high_risk,
                medium_risk=medium_risk,
                low_risk=low_risk,
                recent=recent
            )

    except Exception as e:
        return f"Error: {str(e)}", 500


# MODEL PERFORMANCE ROUTE
@app.route("/model")
def model_performance():
    return render_template("model.html")

# PDF REPORT ROUTE
@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    try:
        # Get data from form
        risk_percentage = request.form.get("risk_percentage")
        risk_level = request.form.get("risk_level")
        recommendation = request.form.get("recommendation")
        shap_plot = request.form.get("shap_plot")

        # Patient details
        patient_data = {
            "Race" : request.form.get("race"),
            "Gender" : request.form.get("gender"),
            "Age Group" : request.form.get("age"),
            "Time in Hospital" : request.form.get("time_in_hospital")  + " days",
            "Medications" : request.form.get("num_medications"),
            "Lab Procedures" : request.form.get("num_lab_procedures"),
            "Diagnoses" : request.form.get("number_diagnoses"),
            "Inpatient Visits" : request.form.get("number_inpatient"),
            "Emergency Visits" : request.form.get("number_emergency"),
            "Primary Diagnosis" : request.form.get("diag_1_group"),
            "Insulin" : request.form.get("insulin"),
            "Total Visits" : request.form.get("total_visits")
        }

        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightmargin=50, leftmargin=50, topmargin=50, bottommargin=50)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#1a237e"),
            spaceAfter=5,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.grey,
            spaceAfter=20,
        )

        story.append(Paragraph("Pakistan Hospital Readmission Predictor", title_style))
        story.append(Paragraph("AI Powered Clinical Support Report", subtitle_style))
        story.append(Spacer(1,10))

        # Risk Score Section
        if risk_level == "High Risk":
            risk_color = colors.HexColor("#c62828")
        elif risk_level == "Medium Risk":
            risk_color = colors.HexColor("#e65100")
        else:
            risk_color = colors.HexColor("#2e7d32")

        risk_style = ParagraphStyle(
            "Risk",
            parent=styles["Normal"],
            fontSize=28,
            textColor=risk_color,
            spaceAfter=5,
            fontName="Helvetica-Bold"
        )    
        risk_label_style = ParagraphStyle(
            "Risk Label",
            parent=styles["Normal"],
            fontSize=14,
            textColor=risk_color,
            spaceAfter=15,
            fontName="Helvetica-Bold"
        )

        story.append(Paragraph(f"Risk Score: {risk_percentage}%", risk_style))
        story.append(Spacer(1, 15))
        story.append(Paragraph(f"Assessment: {risk_level}", risk_label_style))

        # Recommendation box
        rec_style = ParagraphStyle(
            'Rec',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=20,
            leftIndent=10,
            borderPad=10
        )
        story.append(Paragraph(f"Recommendation: {recommendation}", rec_style))
        story.append(Spacer(1, 10))

        # Patient Details Table
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#1565c0'),
            spaceAfter=10
        )
        story.append(Paragraph("Patient Details", section_style))

        table_data = [['Parameter', 'Value']]
        for key, value in patient_data.items():
            table_data.append([key, str(value)])

        table = Table(table_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # SHAP Chart
        story.append(Paragraph("AI Explanation (SHAP)", section_style))
        story.append(Paragraph(
            "The chart below shows which factors influenced this prediction:",
            styles['Normal']
        ))
        story.append(Spacer(1, 10))

        # Decode and add SHAP image
        shap_image_data = base64.b64decode(shap_plot)
        shap_buffer = BytesIO(shap_image_data)
        shap_img = Image(shap_buffer, width=6*inch, height=3.5*inch)
        story.append(shap_img)
        story.append(Spacer(1, 20))

        # Model Info
        story.append(Paragraph("Model Information", section_style))
        model_data = [
            ['Parameter', 'Value'],
            ['Algorithm', 'CatBoost Classifier'],
            ['AUC-ROC Score', '0.6835'],
            ['Recall', '48.39%'],
            ['Training Records', '101,766 patients'],
            ['Decision Threshold', '0.31'],
            ['Explainability', 'SHAP (SHapley Additive exPlanations)'],
        ]
        model_table = Table(model_data, colWidths=[3*inch, 3*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(model_table)
        story.append(Spacer(1, 20))

        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            spaceAfter=10
        )
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI system for clinical decision support only. "
            "It should not replace professional medical judgment. Always consult with qualified "
            "medical professionals before making clinical decisions.",
            disclaimer_style
        ))

        # Build PDF
        doc.build(story)
        buffer.seek(0)

        from flask import send_file
        return send_file(
            buffer,
            as_attachment=True,
            download_name='readmission_report.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        return f"PDF Error: {str(e)}", 500


# RUN APP
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

