"""plurai._mocking — private mock runtime subpackage.

Consolidates the mock HTTP client, local-service launcher, mock-response
resolver, LangGraph interception layer, and session lifecycle helpers into a
coherent private package. The root ``plurai`` serves its public names from here
(see ``plurai._surface``); internal modules import from each other by explicit submodule path.
"""
