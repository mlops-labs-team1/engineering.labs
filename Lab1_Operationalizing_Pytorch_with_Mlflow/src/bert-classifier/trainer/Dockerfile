FROM gcr.io/engineeringlab/base:latest

# Include GCP service account credentials with the build
# Run docker build .... --build-arg GCP_CREDS_JSON_BASE64="xxxx"
# where xxxx is a base64-encoded version of GCP creds JSON
# The lab repo has a secret named GCP_CREDS_JSON_BASE64 with required data
ARG GCP_CREDS_JSON_BASE64
ENV GCP_CREDS_JSON_BASE64_ENV=${GCP_CREDS_JSON_BASE64}
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/gcp.json
RUN echo "${GCP_CREDS_JSON_BASE64_ENV}"| base64 --decode > "${GOOGLE_APPLICATION_CREDENTIALS}"