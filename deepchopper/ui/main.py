import multiprocessing
from functools import partial
from pathlib import Path

import gradio as gr
import lightning
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

import deepchopper
from deepchopper.deepchopper import default, encode_qual, remove_intervals_and_keep_left, smooth_label_region
from deepchopper.models.llm import (
    tokenize_and_align_labels_and_quals,
)
from deepchopper.utils import (
    summary_predict,
)


def parse_fq_record(text: str):
    """Parse a single FASTQ record into a dictionary."""
    lines = text.strip().split("\n")
    for i in range(0, len(lines), 4):
        content = lines[i : i + 4]
        record_id, seq, _, qual = content
        assert len(seq) == len(qual)  # noqa: S101

        yield {
            "id": record_id,
            "seq": seq,
            "qual": encode_qual(qual, default.KMER_SIZE),
            "target": [0, 0],
        }


def load_dataset(text: str, tokenizer):
    """Load dataset from text."""
    dataset = Dataset.from_generator(parse_fq_record, gen_kwargs={"text": text}).with_format("torch")
    tokenized_dataset = dataset.map(
        partial(
            tokenize_and_align_labels_and_quals,
            tokenizer=tokenizer,
            max_length=tokenizer.max_len_single_sentence,
        ),
        num_proc=multiprocessing.cpu_count(),  # type: ignore
    ).remove_columns(["id", "seq", "qual", "target"])
    return dataset, tokenized_dataset


def predict(
    text: str,
    smooth_window_size: int = 21,
    min_interval_size: int = 13,
    approved_interval_number: int = 20,
    max_process_intervals: int = 8,  # default is 4
    batch_size: int = 1,
    num_workers: int = 1,
):
    tokenizer = deepchopper.models.llm.load_tokenizer_from_hyena_model(model_name="hyenadna-small-32k-seqlen")
    dataset, tokenized_dataset = load_dataset(text, tokenizer)

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    model = deepchopper.DeepChopper.from_pretrained("yangliz5/deepchopper")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = lightning.pytorch.trainer.Trainer(
        accelerator=accelerator,
        devices=-1,
        deterministic=False,
        logger=False,
    )

    predicts = trainer.predict(model=model, dataloaders=dataloader, return_predictions=True)

    assert len(predicts) == 1  # noqa: S101

    smooth_interval_json: list[dict[str, int]] = []
    highlighted_text: list[tuple[str, str | None]] = []

    for idx, preds in enumerate(predicts):
        true_prediction, _true_label = summary_predict(predictions=preds[0], labels=preds[1])

        _id = dataset[idx]["id"]
        seq = dataset[idx]["seq"]

        smooth_predict_targets = smooth_label_region(
            true_prediction[0], smooth_window_size, min_interval_size, approved_interval_number
        )

        if not smooth_predict_targets or len(smooth_predict_targets) > max_process_intervals:
            continue

        # zip two consecutive elements
        _selected_seqs, selected_intervals = remove_intervals_and_keep_left(seq, smooth_predict_targets)
        total_intervals = sorted(selected_intervals + smooth_predict_targets)

        smooth_interval_json.extend({"start": i[0], "end": i[1]} for i in smooth_predict_targets)

        highlighted_text.extend(
            (seq[interval[0] : interval[1]], "ada" if interval in smooth_predict_targets else None)
            for interval in total_intervals
        )
    return smooth_interval_json, highlighted_text


def process_input(text: str | None, file: str | None):
    """Process the input and return the prediction."""
    if not text and not file:
        gr.Warning("Both text and file are empty")

    if file:
        MAX_LINES = 4
        file_content = []
        with Path(file).open() as f:
            for idx, line in enumerate(f):
                if idx >= MAX_LINES:
                    break
                file_content.append(line)
        text = "".join(file_content)
        return predict(text=text)

    return predict(text=text)


def create_gradio_app():
    """Create a Gradio app for DeepChopper."""
    example = (
        "@1065:1135|393d635c-64f0-41ed-8531-12174d8efb28+f6a60069-1fcf-4049-8e7c-37523b4e273f\n"
        "GCAGCTATGAATGCAAGGCCACAAGGTGGATGGAAGAGTTGTGGAACCAAAGAGCTGTCTTCCAGAGAAGATTTCGAGATAAGTCGCCCATCAGTGAACAAGATATTGTTGGTGGCATTTGATGAGAACGTTCCAAGATTATTGACAGATTAGTGAAAAGTAAGATTGAAATCATGACTGACCGTAAGTGGCAAGAAAGGGCTTTTGCCTTTGTAACCTTTGACGACCATGACTCCGTGGATAAGATTGTCATTCAGAATACCATACTGTGAATGGCCACATCTTTATTGTGAAGTTAGAAAAGCCCTGTCAAAGCAAGAGATGAATCAGTGCTTCTCCAGCCAAAGAGGTCGAAGTGGTTCTGGAAACTTTGGTGGTGGTCGTGGAGGTGGTTTCGGTGGGAATGACAACTCGGTCGTGGAGGAAACTTCAGTGGTCGTGGTGGCTTTGGTGGCAGCCGTGGTGGTGGTGGATATGGTGGCAGTGGGGATGGCTATAATGGATTTGGTAATGATGGAAGCAATTTGGAGGTGGTGGAAGCTACAATGATTTTGGGAATTACAACAATCAGTCTTCAAATTTTGGACCCCTAGGAGGAAATTTTGGTAGAAGCTCTGGCCCCATGGCGGTGGAGGCCAAATACTTTTGCAAACCACGAAACCAAGGTGGCTATGGCGGTCCAGCAGCAGCAGTAGCTATGGCAGTGGCAGAAGATTTTAATTAGGAAACAAAGCTTAGCAGGAGAGGAGAGCCAGAGAAGTGACAGGGAAGTACAGGTTACAACAGATTTGTGAACTCAGCCCAAGCACAGTGGTGGCAGGGCCTAGCTGCTACAAAGAAGACATGTTTTAGACAAATACTCATGTGTATGGGCAAAACTTGAGGACTGTATTTGTGACTAACTGTATAACAGGTTATTTTAGTTTCTGTTTGTGGAAAGTGTAAAGCATTCCAACAAAGGTTTTTAATGTAGATTTTTTTTTTTGCACCCCATGCTGTTGATTTGCTAAATGTAACAGTCTGATCGTGACGCTGAATAAATGTCTTTTTTAAAAAAAAAAAAAAGCTCCCTCCCATCCCCTGCTGCTAACTGATCCCATTATATCTAACCTGCCCCCCCATATCACCTGCTCCCGAGCTACCTAAGAACAGCTAAAAGAGCACACCCGCATGTAGCAAAATAGTGGGAAGATTATAGGTAGAGGCGACAAACCTACCGAGCCTGGTGATAGCTGGTTGTCCTAGATAGAATCTTAGTTCAACTTTAAATTTGCCCACAGAACCCTCTAAATCCCCTTGTAAATTTAACTGTTAGTCCAAAGAGGAACAGCTCTTTGGACACTAGGAAAAAACCTTGTAGAGAGTAAAAAATCAACACCCA\n"
        "+\n"
        ".0==?SSSSSSSSSSSH2216<868;SSSSSSSSSQQSRSIIHEDDESSSSSSJIKMGEKISSJJICCBDQ?;;8:;,**(&$'+501)\"#$()+%&&0<5+*/('%'))))'''$##\"\"\"\"%&--$\"\"\"('%)1L3*'')'#\"#&+*$&\"\"#*(&'''+,,<;9<BHGF//.LKORQSK<###%*-89<FSSSSE=BAFHFDB???3313NN?>=ANOSJDCADHGMOQSSD=7>BRRSPIEEEOQSSQ4->LIC7EE045///03IIJQSSSNGE6('.5??@A@=,,EGRSPKJ<==<556GFLLQRANSSSSSSSSG...*%%%(***(%'3@LOOSSSSM...7BCMMSSSSSSSSSSSSSSSDFIPSSSGGGGPOQLIHIL4103HMSILLNOSSSSSSSSSS22CBCGSHHHHSSSSSSSSD??@<<<:DDDSSSSSSSSSSA@6688OSSSSSROJJKLSNNNMSSSSQPOOSOOQSSSSSRRHIHISSRSSSSSSSSSSSJFF=??@SSQRK:424<444FFG///1S@@@ASNNNNPN:4JMDDLPSSSSSSBA?B?@@+'&'BD**8EDEFQPIMLE$$&',79CSJJPSGA+***DN;3-('&(;>6(()/-,,)%')1FRNNJ-:=>GC;&;CHNFFDCEEKJLFA22/27A.....HSQLHL))8<=?JSSSFGSKIHDDCCEFDAA@CFJKLNL>:9/1>>?OSLK@+HPSA;>>>K;;;;SSSSOQLPPMORSSSSSQSSSSSSS=:9**?D889SSRFFEDKJJJEEDKSSSNNOSSS.---,&*++SSSSQRSSSSQPGED<<89<@GJ999:SSKBBBAJHK=SSSJJKNMGHKKHQA<<>OPKFEAACDHJKMORB/)'((6**)15DA99;JSQSSS2())+J))EGMQOMMKJF>?<<AA620..D..,/112SOIIJSQFNEEEOMF?066=>@4,3;B>87FSSSSSSSSSSSSSSS<<::5658@AHMMSSRECC448/=<<>SSCB:5546;<??KF==;;FFEDFHKKJG):C>=>BJHINJFDPPPPPPPPPPPPPP%'*%$%+-%'(-22&&%('''&&&#\"\"%&'+0,,0;:1&\"\"%'(+++8'**(\"$$#&$'**//.3497$\"3CFHLOSSSSR:887:;;FSSRPRSSS4433$#$%&$$-056>@:;>=@?AHEFEC;*EKMSSRSRRDB>=AFRSSSSBSOOPSMDAABHH976951-9DHPQO/---?@ELSSQSRJHKKBKKLSSLINSOSSQSRIMSSSSSS>?MKIINSSGSSSSSSSQQMK544MJKKNKHGGLFFGBDB?EHIKGD?@DHPPIIF555)&(+,ADSSSSRQSSSQSS=9/0JJMSQSOSSO/97=B@=:>"
    )

    custom_css = """
    .header { text-align: center; margin-bottom: 30px; }
    .footer { text-align: center; margin-top: 30px; font-size: 0.8em; color: #666; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML(
            """
            <div class="header">
                <h1>🧬 DeepChopper: DNA Sequence Analysis</h1>
                <p>Analyze DNA sequences and detect artificial sequences</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Input DNA Sequence", placeholder="Paste your DNA sequence here...", lines=10
                )
                file_input = gr.File(label="Or upload a FASTQ file")
                submit_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                json_output = gr.JSON(label="Detected Artificial Regions")
                highlighted_text = gr.HighlightedText(label="Highlighted Sequence")

        submit_btn.click(fn=process_input, inputs=[text_input, file_input], outputs=[json_output, highlighted_text])

        gr.Examples(
            examples=[[example]],
            inputs=[text_input],
        )

        gr.HTML(
            """
            <div class="footer">
                <p>DeepChopper - Powered by AI for DNA sequence analysis</p>
            </div>
            """
        )

    return demo


def main():
    """Launch the Gradio app."""
    app = create_gradio_app()
    app.launch()


if __name__ == "__main__":
    main()
