from setuptools import setup, find_packages

setup(
    name="newsgator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "feedparser>=6.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "scikit-learn>=1.2.0",
        "openai>=1.0.0",
        "jinja2>=3.1.0",
        "markdown>=3.4.0",
        "markdown2>=2.4.0",
    ],
    entry_points={
        "console_scripts": [
            "newsgator=newsgator:main",
        ],
    },
    python_requires=">=3.8",
    author="Newsgator Team",
    description="RSS feed aggregator with content similarity analysis, LLM translation to Italian, and newspaper-styled HTML publishing",
    keywords="rss, news, aggregator, translation, nlp",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
