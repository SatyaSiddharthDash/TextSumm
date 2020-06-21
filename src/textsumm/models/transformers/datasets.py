from torch.utils.data import Dataset, IterableDataset
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import itertools
import jsonlines


def _line_iter(file_path):
    with open(file_path, "r", encoding="utf8") as fd:
        for line in fd:
            yield line


def _preprocess(sentences, preprocess_pipeline, word_tokenize=None):
    """
    Helper function to preprocess a list of paragraphs.
    Args:
        param (Tuple): params are tuple of (a list of strings,
            a list of preprocessing functions, and function to tokenize
            setences into words). A paragraph is represented with a
            single string with multiple setnences.
    Returns:
        list of list of strings, where each string is a token or word.
    """
    if preprocess_pipeline is not None:
        for function in preprocess_pipeline:
            sentences = function(sentences)

    if word_tokenize is None:
        return sentences
    else:
        return sentences, [word_tokenize(sentence) for sentence in sentences]


def _create_data_from_iterator(iterator, preprocessing, word_tokenize):
    for line in iterator:
        yield _preprocess(
            sentences=line,
            preprocess_pipeline=preprocessing,
            word_tokenize=word_tokenize,
        )


class IterableSummarizationDataset(IterableDataset):
    def __init__(
        self,
        source_file,
        target_file=None,
        source_preprocessing=None,
        target_preprocessing=None,
        word_tokenization=None,
        top_n=-1,
    ):
        """
        Create a summarization dataset instance given the
        paths of the source file and the target file
        Args:
            source_file (str): Full path of the file which contains a list of
                the paragraphs with line break as seperator.
            target_file (str): Full path of the file which contains a list of
                the summaries for the paragraphs in the source file with line break as
                seperator.
            source_preprocessing (list of functions): A list of preprocessing functions
                to process the paragraphs in the source file.
            target_preprocessing (list of functions): A list of preprocessing functions
                to process the paragraphs in the source file.
            word_tokenization (function): Tokenization function for tokenize the
                paragraphs and summaries. The tokenization method is used for sentence
                selection in
                :meth:`utils_nlp.models.transformers.extractive_summarization.
                ExtSumProcessor.preprocess`
            top_n (int, optional): The number which specifies how many examples in the
                beginning of the paragraph and summary lists that will be processed by
                this function. Defaults to -1, which means the whole lists of paragraphs
                and summaries should be procsssed.
        """

        source_iter = _line_iter(source_file)

        if top_n != -1:
            source_iter = itertools.islice(source_iter, top_n)

        self._source = _create_data_from_iterator(
            source_iter, source_preprocessing, word_tokenization
        )

        if target_file:
            target_iter = _line_iter(target_file)
            if top_n != -1:
                target_iter = itertools.islice(target_iter, top_n)
            self._target = _create_data_from_iterator(
                target_iter, target_preprocessing, word_tokenization
            )
        else:
            self._target = None

    def __iter__(self):
        for x in self._source:
            yield x

    def get_source(self):
        return self._source

    def get_target(self):
        return self._target


class SummarizationDataset(Dataset):
    def __init__(
        self,
        source_file,
        source=None,
        target_file=None,
        target=None,
        source_preprocessing=None,
        target_preprocessing=None,
        word_tokenize=None,
        top_n=-1,
        n_processes=-1,
    ):
        """
        Create a summarization dataset instance given the
        paths of the source file and the target file.
        Args:
            source_file (str): Full path of the file which contains a list of
                the input paragraphs with line break as seperator.
            source (list of str, optional): a list of input paragraphs.
                Defaults to None.
            target_file (str, optional): Full path of the file which contains a list of
                the summaries for the paragraphs in the source file with line break
                as seperator.
            target (list of str, optional): a list of summaries correponding to
                `source`. Defaults to None.
            source_preprocessing (list of functions): A list of preprocessing functions
                to process the paragraphs in the source file.
            target_preprocessing (list of functions): A list of preprocessing functions
                to process the summaries in the target file.
            top_n (int, optional): Number of examples to load from the input files.
                Defaults to -1, which means the entire dataset is loaded.
            n_processes (int, optional): Number of CPUs to use to process the data in
                parallel. Defaults to -1, which means all the CPUs will be used.
        """
        self._source_txt = []
        if source_file is not None and os.path.exists(source_file):
            with open(source_file, encoding="utf-8") as f:
                if top_n != -1:
                    self._source_txt = list(itertools.islice(f, top_n))
                else:
                    self._source_txt = f.readlines()
        if source:
            self._source_txt.extend(source)

        self._target_txt = []
        if target_file is not None and os.path.exists(target_file):
            with open(target_file, encoding="utf-8") as f:
                if top_n != -1:
                    self._target_txt = list(itertools.islice(f, top_n))
                else:
                    self._target_txt = f.readlines()
        if target:
            self._target_txt.extend(target)

        if len(self._target_txt) == 0:
            self._target_txt = None
        else:
            assert len(self._source_txt) == len(self._target_txt)

        result = parallel_preprocess(
            self._source_txt,
            preprocess_pipeline=source_preprocessing,
            word_tokenize=word_tokenize,
            num_pool=n_processes,
        )
        if word_tokenize:
            self._source_txt = list(
                map(lambda x: x[0], filter(lambda x: len(x[0]) > 0, result))
            )
            self._source = list(
                map(lambda x: x[1], filter(lambda x: len(x[1]) > 0, result))
            )
        else:
            self._source = list(
                map(lambda x: x, filter(lambda x: len(x) > 0, result))
            )

        if self._target_txt is not None and len(self._target_txt) > 0:
            result = parallel_preprocess(
                self._target_txt,
                preprocess_pipeline=target_preprocessing,
                word_tokenize=word_tokenize,
                num_pool=n_processes,
            )

            if word_tokenize:
                self._target_txt = list(
                    map(lambda x: x[0], filter(
                        lambda x: len(x[0]) > 0, result))
                )
                self._target = list(
                    map(lambda x: x[1], filter(
                        lambda x: len(x[1]) > 0, result))
                )
            else:
                self._target = list(
                    map(lambda x: x, filter(lambda x: len(x) > 0, result))
                )

    def shorten(self, top_n=None):
        if top_n is None:
            return self
        elif top_n <= len(self._source):
            self._source = self._source[0:top_n]
            self._source_txt = self._source_txt[0:top_n]

            if self._target_txt is not None:
                self._target = self._target[0:top_n]
                self._target_txt = self._target_txt[0:top_n]
            return self
        else:
            return self

    def __getitem__(self, idx):
        # tupe is more adaptive
        if self._target_txt is None:
            return {"src": self._source[idx], "src_txt": self._source_txt[idx]}
        else:
            return {
                "src": self._source[idx],
                "src_txt": self._source_txt[idx],
                "tgt": self._target[idx],
                "tgt_txt": self._target_txt[idx],
            }

    def __len__(self):
        return len(self._source)

    def get_source(self):
        return self._source

    def get_source_txt(self):
        return self._source_txt

    def get_target_txt(self):
        return self._target_txt

    def get_target(self):
        return self._target

    def save_to_jsonl(self, output_file):
        with jsonlines.open(output_file, mode="w") as writer:
            if self._target_txt is None:
                for src in self._source:
                    writer.write({"src": src})
            else:
                for src, tgt in zip(self._source, self._target):
                    writer.write({"src": src, "tgt": tgt})


def _preprocess(sentences, preprocess_pipeline, word_tokenize=None):
    """
    Helper function to preprocess a list of paragraphs.
    Args:
        param (Tuple): params are tuple of (a list of strings,
            a list of preprocessing functions, and function to tokenize
            setences into words). A paragraph is represented with a
            single string with multiple setnences.
    Returns:
        list of list of strings, where each string is a token or word.
    """
    if preprocess_pipeline is not None:
        for function in preprocess_pipeline:
            sentences = function(sentences)

    if word_tokenize is None:
        return sentences
    else:
        return sentences, [word_tokenize(sentence) for sentence in sentences]


def parallel_preprocess(
    input_data, preprocess_pipeline, word_tokenize=None, num_pool=-1
):
    """
    Process data in parallel using multiple CPUs.
    Args:
        input_data (list): List if input strings to process.
        preprocess_pipeline (list): List of functions to apply on the input data.
        word_tokenize (func, optional): A tokenization function used to tokenize
            the results from preprocess_pipeline.
        num_pool (int, optional): Number of CPUs to use. Defaults to -1 and all
            available CPUs are used.
    Returns:
        list: list of processed text strings.
    """
    if num_pool == -1:
        num_pool = cpu_count()

    num_pool = min(num_pool, len(input_data))

    p = Pool(num_pool)

    results = p.map(
        partial(
            _preprocess,
            preprocess_pipeline=preprocess_pipeline,
            word_tokenize=word_tokenize,
        ),
        input_data,
        chunksize=min(1, int(len(input_data) / num_pool)),
    )
    p.close()
    p.join()

    return results
