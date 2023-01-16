docker run --gpus all --rm -it -v /media/walter/Storage/Projects/flip-sample-application/data:/data --entrypoint bash nvflare-in-one

bash start_nvflare_components.sh &
./poc/admin/startup/fl_admin.sh

upload_app aekl-app
set_run_number 1
deploy_app aekl-app all
start_app all

upload_app ddpm-unconditioned-app
set_run_number 1
deploy_app ddpm-unconditioned-app all
start_app all