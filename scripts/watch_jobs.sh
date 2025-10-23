#!/bin/bash

watch -n2 'squeue -u $USER -o "%.18i %.12j %.2t %.10M %.6D %R"'
