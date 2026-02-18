{{- define "chart.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{- define "chart.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}

{{- define "chart.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{- define "chart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "chart.labels" -}}
helm.sh/chart: {{ include "chart.chart" . }}
{{ include "chart.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Common environment variables for the API container.
Provides MLflow tracking, S3 artifact access, and app configuration.
*/}}
{{- define "course-mlops.commonEnv" -}}
- name: MLFLOW_TRACKING_URI
  value: http://{{ .Release.Name }}-mlflow:{{ .Values.mlflow.service.port }}
- name: MLFLOW_S3_ENDPOINT_URL
  value: http://{{ .Release.Name }}-minio:9000
- name: AWS_ACCESS_KEY_ID
  value: {{ .Values.minio.rootUser }}
- name: AWS_SECRET_ACCESS_KEY
  value: {{ .Values.minio.rootPassword }}
- name: CML_FAIL_FAST
  value: {{ .Values.api.env.failFast | quote }}
- name: CML_MODEL_STRATEGY
  value: {{ .Values.api.env.modelStrategy | quote }}
- name: CML_LOG_LEVEL
  value: {{ .Values.api.env.logLevel | quote }}
- name: CML_LOG_FORMAT
  value: {{ .Values.api.env.logFormat | quote }}
- name: CML_DB_HOST
  value: {{ include "chart.fullname" . }}-postgresql-rw
- name: CML_DB_PORT
  value: "5432"
- name: CML_DB_USER
  valueFrom:
    secretKeyRef:
      name: {{ include "chart.fullname" . }}-postgresql-app
      key: username
- name: CML_DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "chart.fullname" . }}-postgresql-app
      key: password
- name: CML_DB_NAME
  value: {{ .Values.postgresql.database }}
{{- end -}}
