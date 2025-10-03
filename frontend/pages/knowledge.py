"""
Knowledge Base Explorer for Nutritional AI Chatbot
Search, visualize, and explore the nutrition knowledge graph and vector database
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Import our main app's API client
from streamlit_app import APIClient, API_BASE_URL


class KnowledgeManager:
    """Handle knowledge base operations"""

    @staticmethod
    def search_knowledge_base(
        query: str, search_type: str = "hybrid", limit: int = 10
    ) -> Dict[str, Any]:
        """Search the knowledge base"""
        try:
            params = {
                "query": query,
                "search_type": search_type,
                "limit": limit,
                "include_metadata": True,
            }

            response = requests.get(
                f"{API_BASE_URL}/api/knowledge/search", params=params, timeout=20
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def get_entities(entity_type: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get entities from the knowledge graph"""
        try:
            params = {"limit": limit}
            if entity_type:
                params["entity_type"] = entity_type

            response = requests.get(
                f"{API_BASE_URL}/api/knowledge/entities", params=params, timeout=15
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def get_graph_data(center_node: str = None, depth: int = 2) -> Dict[str, Any]:
        """Get graph visualization data"""
        try:
            params = {"depth": depth}
            if center_node:
                params["center_node"] = center_node

            response = requests.get(
                f"{API_BASE_URL}/api/knowledge/graph", params=params, timeout=20
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}

    @staticmethod
    def analyze_text(text: str) -> Dict[str, Any]:
        """Analyze text for entities and relationships"""
        try:
            payload = {"text": text}

            response = requests.post(
                f"{API_BASE_URL}/api/knowledge/analyze", json=payload, timeout=15
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Connection error: {str(e)}"}


def show_search_interface():
    """Display the search interface"""
    st.subheader("🔍 Search Knowledge Base")

    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="e.g., foods high in protein, vitamin C sources, Mediterranean diet",
                help="Search for foods, nutrients, health conditions, or dietary information",
            )

        with col2:
            search_type = st.selectbox(
                "Search Type",
                ["hybrid", "vector", "graph", "text"],
                help="Hybrid combines multiple search methods for best results",
            )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            search_button = st.form_submit_button("Search 🔍", type="primary")

        with col2:
            limit = st.number_input("Results", min_value=5, max_value=50, value=10)

    if search_button and query.strip():
        with st.spinner("Searching knowledge base..."):
            result = KnowledgeManager.search_knowledge_base(
                query.strip(), search_type, limit
            )

        if result["success"]:
            search_data = result["data"]
            results = search_data.get("results", [])

            if results:
                st.success(f"Found {len(results)} results")

                # Display search metadata
                metadata = search_data.get("metadata", {})
                if metadata:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Search Time", f"{metadata.get('search_time', 'N/A')}s"
                        )
                    with col2:
                        st.metric("Total Matches", metadata.get("total_matches", "N/A"))
                    with col3:
                        st.metric("Search Type", search_type.title())

                # Display results
                for i, item in enumerate(results, 1):
                    with st.expander(
                        f"📄 Result {i}: {item.get('title', 'Untitled')}",
                        expanded=(i == 1),
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(
                                f"**Content:** {item.get('content', 'No content available')}"
                            )

                            if item.get("summary"):
                                st.markdown(f"**Summary:** {item['summary']}")

                        with col2:
                            if item.get("score"):
                                score = float(item["score"])
                                st.metric("Relevance Score", f"{score:.3f}")

                            if item.get("source"):
                                st.markdown(f"**Source:** {item['source']}")

                            if item.get("entity_type"):
                                st.markdown(f"**Type:** {item['entity_type']}")

                        # Show related entities or relationships
                        if item.get("related_entities"):
                            st.markdown("**Related Entities:**")
                            related_cols = st.columns(
                                min(len(item["related_entities"]), 4)
                            )
                            for j, entity in enumerate(item["related_entities"][:4]):
                                with related_cols[j]:
                                    st.markdown(f"• {entity}")
            else:
                st.warning("No results found. Try different keywords or search type.")

        else:
            st.error(f"Search failed: {result['error']}")


def show_entity_explorer():
    """Display the entity explorer interface"""
    st.subheader("🏷️ Entity Explorer")

    col1, col2 = st.columns([1, 1])

    with col1:
        entity_type = st.selectbox(
            "Entity Type",
            [
                "All",
                "Food",
                "Nutrient",
                "Health_Condition",
                "Culture",
                "Age_Group",
                "Location",
            ],
            help="Filter entities by type",
        )

    with col2:
        limit = st.number_input(
            "Number of Entities", min_value=10, max_value=200, value=50
        )

    if st.button("Load Entities", type="primary"):
        with st.spinner("Loading entities..."):
            entity_type_param = None if entity_type == "All" else entity_type
            result = KnowledgeManager.get_entities(entity_type_param, limit)

        if result["success"]:
            entities = result["data"].get("entities", [])

            if entities:
                st.success(f"Loaded {len(entities)} entities")

                # Create DataFrame for better display
                entity_data = []
                for entity in entities:
                    entity_data.append(
                        {
                            "Name": entity.get("name", "Unknown"),
                            "Type": entity.get("type", "Unknown"),
                            "Properties": len(entity.get("properties", {})),
                            "Relationships": entity.get("relationship_count", 0),
                        }
                    )

                df = pd.DataFrame(entity_data)

                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entities", len(df))
                with col2:
                    st.metric("Entity Types", df["Type"].nunique())
                with col3:
                    st.metric("Avg Properties", f"{df['Properties'].mean():.1f}")
                with col4:
                    st.metric("Total Relationships", df["Relationships"].sum())

                # Type distribution chart
                if len(df) > 0:
                    fig = px.pie(
                        df, names="Type", title="Entity Type Distribution", hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Entity table with search
                st.subheader("Entity Details")
                search_term = st.text_input(
                    "Search entities:", placeholder="Type to filter entities..."
                )

                if search_term:
                    filtered_df = df[
                        df["Name"].str.contains(search_term, case=False, na=False)
                    ]
                else:
                    filtered_df = df

                st.dataframe(filtered_df, use_container_width=True, height=400)

                # Entity details
                if not filtered_df.empty:
                    selected_entity = st.selectbox(
                        "Select entity for details:",
                        options=filtered_df["Name"].tolist(),
                    )

                    if selected_entity:
                        entity_details = next(
                            (e for e in entities if e.get("name") == selected_entity),
                            None,
                        )

                        if entity_details:
                            st.subheader(f"Details: {selected_entity}")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown(
                                    f"**Type:** {entity_details.get('type', 'Unknown')}"
                                )

                                properties = entity_details.get("properties", {})
                                if properties:
                                    st.markdown("**Properties:**")
                                    for key, value in properties.items():
                                        st.markdown(f"• **{key}:** {value}")

                            with col2:
                                relationships = entity_details.get("relationships", [])
                                if relationships:
                                    st.markdown("**Relationships:**")
                                    for rel in relationships[:10]:  # Show first 10
                                        rel_type = rel.get("type", "RELATED_TO")
                                        target = rel.get("target", "Unknown")
                                        st.markdown(f"• {rel_type} → {target}")
            else:
                st.warning("No entities found.")

        else:
            st.error(f"Failed to load entities: {result['error']}")


def show_graph_visualization():
    """Display graph visualization interface"""
    st.subheader("🌐 Knowledge Graph Visualization")

    col1, col2 = st.columns([2, 1])

    with col1:
        center_node = st.text_input(
            "Center Node (optional)",
            placeholder="e.g., Protein, Vitamin C, Mediterranean Diet",
            help="Specify a node to center the visualization around",
        )

    with col2:
        depth = st.number_input("Graph Depth", min_value=1, max_value=3, value=2)

    if st.button("Generate Graph", type="primary"):
        with st.spinner("Building graph visualization..."):
            result = KnowledgeManager.get_graph_data(center_node or None, depth)

        if result["success"]:
            graph_data = result["data"]
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

            if nodes and edges:
                st.success(f"Graph loaded: {len(nodes)} nodes, {len(edges)} edges")

                # Create network graph using networkx and plotly
                G = nx.Graph()

                # Add nodes
                for node in nodes:
                    G.add_node(
                        node["id"],
                        label=node.get("label", node["id"]),
                        type=node.get("type", "Unknown"),
                        size=node.get("size", 10),
                    )

                # Add edges
                for edge in edges:
                    G.add_edge(
                        edge["source"],
                        edge["target"],
                        relation=edge.get("relation", "CONNECTED"),
                    )

                # Calculate layout
                pos = nx.spring_layout(G, k=1, iterations=50)

                # Create plotly figure
                fig = go.Figure()

                # Add edges
                edge_x = []
                edge_y = []
                edge_info = []

                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_info.append(
                        G.edges[edge[0], edge[1]].get("relation", "CONNECTED")
                    )

                fig.add_trace(
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        line=dict(width=1, color="#888"),
                        hoverinfo="none",
                        mode="lines",
                        showlegend=False,
                    )
                )

                # Add nodes
                node_x = []
                node_y = []
                node_text = []
                node_colors = []
                node_sizes = []

                color_map = {
                    "Food": "#FF6B6B",
                    "Nutrient": "#4ECDC4",
                    "Health_Condition": "#45B7D1",
                    "Culture": "#96CEB4",
                    "Age_Group": "#FFEAA7",
                    "Location": "#DDA0DD",
                    "Unknown": "#95A5A6",
                }

                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                    node_info = G.nodes[node]
                    node_text.append(
                        f"<b>{node_info['label']}</b><br>Type: {node_info['type']}"
                    )
                    node_colors.append(
                        color_map.get(node_info["type"], color_map["Unknown"])
                    )
                    node_sizes.append(max(10, node_info.get("size", 10)))

                fig.add_trace(
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        hovertemplate="%{text}<extra></extra>",
                        text=[G.nodes[node]["label"] for node in G.nodes()],
                        textposition="middle center",
                        marker=dict(
                            size=node_sizes,
                            color=node_colors,
                            line=dict(width=2, color="white"),
                        ),
                        showlegend=False,
                    )
                )

                fig.update_layout(
                    title="Knowledge Graph Visualization",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Hover over nodes for details",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002,
                            xanchor="left",
                            yanchor="bottom",
                            font=dict(color="#888", size=12),
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Graph statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nodes", len(nodes))
                with col2:
                    st.metric("Edges", len(edges))
                with col3:
                    st.metric("Density", f"{nx.density(G):.3f}")
                with col4:
                    st.metric(
                        "Avg Degree",
                        f"{sum(dict(G.degree()).values())/len(G.nodes()):.1f}",
                    )

                # Legend
                st.subheader("Node Types")
                legend_cols = st.columns(len(color_map))
                for i, (node_type, color) in enumerate(color_map.items()):
                    with legend_cols[i % len(legend_cols)]:
                        st.markdown(f"🔴 **{node_type}**", unsafe_allow_html=True)

            else:
                st.warning(
                    "No graph data available. Try different parameters or check if the knowledge base is populated."
                )

        else:
            st.error(f"Failed to load graph: {result['error']}")


def show_text_analysis():
    """Display text analysis interface"""
    st.subheader("📝 Text Analysis")

    st.markdown(
        """
    Analyze any text to extract nutritional entities, relationships, and insights.
    This tool uses NLP to identify foods, nutrients, health conditions, and their relationships.
    """
    )

    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Example: I want to eat foods rich in protein and vitamin C to boost my immune system and build muscle.",
        height=150,
        help="Enter any text related to nutrition, food, or health",
    )

    if st.button("Analyze Text", type="primary") and text_input.strip():
        with st.spinner("Analyzing text..."):
            result = KnowledgeManager.analyze_text(text_input.strip())

        if result["success"]:
            analysis = result["data"]

            # Display entities
            if analysis.get("entities"):
                st.subheader("🏷️ Extracted Entities")

                entities_by_type = {}
                for entity in analysis["entities"]:
                    entity_type = entity.get("type", "Unknown")
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity)

                cols = st.columns(min(len(entities_by_type), 4))
                for i, (entity_type, entities) in enumerate(entities_by_type.items()):
                    with cols[i % len(cols)]:
                        st.markdown(f"**{entity_type}:**")
                        for entity in entities:
                            confidence = entity.get("confidence", 0.0)
                            st.markdown(f"• {entity['text']} ({confidence:.2f})")

            # Display relationships
            if analysis.get("relationships"):
                st.subheader("🔗 Identified Relationships")

                for rel in analysis["relationships"]:
                    source = rel.get("source", "Unknown")
                    relation = rel.get("relation", "RELATED_TO")
                    target = rel.get("target", "Unknown")
                    confidence = rel.get("confidence", 0.0)

                    st.markdown(
                        f"• **{source}** → {relation} → **{target}** (confidence: {confidence:.2f})"
                    )

            # Display insights
            if analysis.get("insights"):
                st.subheader("💡 Insights")
                for insight in analysis["insights"]:
                    st.info(insight)

            # Display nutritional analysis
            if analysis.get("nutritional_analysis"):
                st.subheader("🥗 Nutritional Analysis")
                nutrition = analysis["nutritional_analysis"]

                col1, col2 = st.columns(2)

                with col1:
                    if nutrition.get("food_items"):
                        st.markdown("**Foods Mentioned:**")
                        for food in nutrition["food_items"]:
                            st.markdown(f"• {food}")

                with col2:
                    if nutrition.get("nutrients"):
                        st.markdown("**Nutrients Discussed:**")
                        for nutrient in nutrition["nutrients"]:
                            st.markdown(f"• {nutrient}")

        else:
            st.error(f"Analysis failed: {result['error']}")


def render_knowledge_page():
    """Main function to render the knowledge base page"""
    st.title("🔍 Knowledge Base Explorer")

    st.markdown(
        """
    Explore our comprehensive nutrition knowledge base with advanced search capabilities,
    entity exploration, graph visualization, and text analysis tools.
    """
    )

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Search", "🏷️ Entities", "🌐 Graph", "📝 Analysis"]
    )

    with tab1:
        show_search_interface()

    with tab2:
        show_entity_explorer()

    with tab3:
        show_graph_visualization()

    with tab4:
        show_text_analysis()
