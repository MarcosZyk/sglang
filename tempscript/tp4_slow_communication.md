# Steps for Reproduction

## 1. Prepare Env

On gnr1 - gnr4:
1. activate python virtual environment
```shell
conda activate sglamx-dong1
```
2. go to the sglang dir
```shell
cd /home/dchen/sglang-amx
```
3. make sure to be at branch ***newbase***
```shell
git checkout newbase
```

## 2. Run SGLang

In ```/home/dchen/sglang-amx``` and on gnr1 - gnr4:

```shell
bash ../scripts/test_tp4.sh
```

## 3. Benchmark SGLang

On gnr1 and in ```/home/dchen/scripts```:

```shell
bash bm_tp4.sh
```

## 4. Profile SGLang

During benchmarking SGLang process and in ```/home/dchen/scripts```:

```shell
bash profile_sgl.sh
```

## 5. Get Result

Step4 will print the dir name of profiled log and the monitored performance metric in console.

Get the dir name, in timestamp format like 1760866678.270784, and go to the dir:

```shell
cd /home/dchen/profile_log/<name>
```
In the dir, run:
```shell
bash ../cp_file.sh <name>-TP-0.trace.json.gz
```

In your own browser, visit:

```
https://morphling-prod.oss-cn-hangzhou-zmf.aliyuncs.com/software/ossutil64/<name>-TP-0.trace.json.gz
```
Then unzip the downloaded gz file and open the json in the following page:

```
https://ui.perfetto.dev/
```


