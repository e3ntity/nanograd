//! Utilities for exporting a [`Scalar`] computation graph as an interactive
//! D3.js visualisation.
//!
//! The main entry-point is [`dump_graph`], which writes a fully self-contained
//! HTML file that can be opened in any modern browser.
use crate::scalar::Scalar;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

/// Helper enum that represents either a data node (`Scalar`) or an operation
/// node while we serialise the graph for visualisation.
#[derive(Debug, Clone)]
enum NodeType {
    Scalar { id: usize, scalar: Scalar },
    Operation { id: usize, op_name: String },
}

impl NodeType {
    fn get_id(&self) -> usize {
        match self {
            NodeType::Scalar { id, .. } => *id,
            NodeType::Operation { id, .. } => *id,
        }
    }
}

/// Render the graph rooted at `out` into `file` as a self-contained HTML
/// document.
///
/// Internally the function walks the graph, converts it into a flat list of
/// nodes/edges and embeds it into a small D3.js application.  No external
/// resources are required besides the D3 CDN.
pub fn dump_graph(out: &Scalar, file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut visited = HashSet::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut node_counter = 0;

    collect_nodes_and_edges(out, &mut visited, &mut nodes, &mut edges, &mut node_counter);

    let leaf_nodes: HashSet<usize> = nodes
        .iter()
        .filter(|node| match node {
            NodeType::Scalar { scalar, .. } => scalar.get_children().is_empty(),
            NodeType::Operation { .. } => false,
        })
        .map(|node| node.get_id())
        .collect();
    let root_node = out.get_arc_ptr();

    // HTML skeleton
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<title>Computational Graph</title>\n");
    html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
    html.push_str("<style>\n");
    html.push_str(
        "body { font-family: Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }\n",
    );
    html.push_str("h1 { text-align: center; color: #333; }\n");
    html.push_str(
        "svg { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }\n",
    );
    html.push_str(".node circle { stroke: #fff; stroke-width: 2px; cursor: pointer; }\n");
    html.push_str(".node rect { stroke: #fff; stroke-width: 2px; cursor: pointer; }\n");
    html.push_str(
        ".node circle.no-grad { stroke: #999; stroke-width: 3px; stroke-dasharray: 5,5; }\n",
    );
    html.push_str(
        ".node text { font-size: 12px; text-anchor: middle; fill: #333; pointer-events: none; }\n",
    );
    html.push_str(".link { stroke: #666; stroke-opacity: 0.6; stroke-width: 2px; fill: none; marker-end: url(#arrow); }\n");
    html.push_str(".root { fill: #e74c3c; }\n");
    html.push_str(".leaf { fill: #27ae60; }\n");
    html.push_str(".internal { fill: #3498db; }\n");
    html.push_str(".operation { fill: #f39c12; }\n");
    html.push_str(".legend { font-size: 14px; }\n");
    html.push_str("</style>\n");
    html.push_str("</head>\n<body>\n");
    html.push_str("<h1>Computational Graph Visualization</h1>\n");
    html.push_str("<svg width=\"1200\" height=\"800\" id=\"graph\"></svg>\n");

    // JavaScript & D3 setup
    html.push_str("<script>\n");
    html.push_str("const svg = d3.select('#graph');\n");
    html.push_str("const width = 1200, height = 800;\n");

    html.push_str("svg.append('defs').append('marker')\n");
    html.push_str("   .attr('id', 'arrow')\n");
    html.push_str("   .attr('viewBox', '0 -5 10 10')\n");
    html.push_str("   .attr('refX', 15)\n");
    html.push_str("   .attr('refY', 0)\n");
    html.push_str("   .attr('markerWidth', 6)\n");
    html.push_str("   .attr('markerHeight', 6)\n");
    html.push_str("   .attr('orient', 'auto')\n");
    html.push_str("   .append('path')\n");
    html.push_str("   .attr('d', 'M0,-5L10,0L0,5')\n");
    html.push_str("   .attr('fill', '#666');\n");

    // Serialize nodes
    html.push_str("const nodes = [\n");
    for (i, node) in nodes.iter().enumerate() {
        match node {
            NodeType::Scalar { id, scalar } => {
                let value = scalar.get_value();
                let grad_opt = scalar.get_grad();
                let has_grad = scalar.has_grad();

                let node_type = if *id == root_node {
                    "root"
                } else if leaf_nodes.contains(id) {
                    "leaf"
                } else {
                    "internal"
                };

                let radius = if *id == root_node {
                    25
                } else if leaf_nodes.contains(id) {
                    20
                } else {
                    15
                };

                let grad_display = match grad_opt {
                    Some(grad) => format!("{:.3}", grad),
                    None => "N/A".to_string(),
                };

                html.push_str(&format!(
                    "  {{ id: {}, value: {:.3}, grad: '{}', hasGrad: {}, type: '{}', radius: {}, shape: 'circle', label: '' }}",
                    id, value, grad_display, has_grad, node_type, radius
                ));
            }
            NodeType::Operation { id, op_name, .. } => {
                html.push_str(&format!(
                    "  {{ id: {}, value: 0, grad: 'N/A', hasGrad: false, type: 'operation', radius: 15, shape: 'rect', label: '{}' }}",
                    id, op_name
                ));
            }
        }
        if i < nodes.len() - 1 {
            html.push_str(",");
        }
        html.push_str("\n");
    }
    html.push_str("];\n");

    // Serialize edges
    html.push_str("const links = [\n");
    for (i, (source_id, target_id)) in edges.iter().enumerate() {
        html.push_str(&format!(
            "  {{ source: {}, target: {} }}",
            source_id, target_id
        ));
        if i < edges.len() - 1 {
            html.push_str(",");
        }
        html.push_str("\n");
    }
    html.push_str("];\n");

    // Zoom behaviour
    html.push_str("const zoom = d3.zoom()\n");
    html.push_str("  .scaleExtent([0.1, 4])\n");
    html.push_str("  .on('zoom', (event) => {\n");
    html.push_str("    container.attr('transform', event.transform);\n");
    html.push_str("  });\n");

    html.push_str("svg.call(zoom);\n");

    // Main container
    html.push_str("const container = svg.append('g');\n");

    // Force-directed layout
    html.push_str("const simulation = d3.forceSimulation(nodes)\n");
    html.push_str("  .force('link', d3.forceLink(links).id(d => d.id).distance(80))\n");
    html.push_str("  .force('charge', d3.forceManyBody().strength(-400))\n");
    html.push_str("  .force('center', d3.forceCenter(width / 2, height / 2));\n");

    // Links
    html.push_str("const link = container.append('g')\n");
    html.push_str("  .selectAll('.link')\n");
    html.push_str("  .data(links)\n");
    html.push_str("  .enter().append('line')\n");
    html.push_str("  .attr('class', 'link');\n");

    // Nodes
    html.push_str("const node = container.append('g')\n");
    html.push_str("  .selectAll('.node')\n");
    html.push_str("  .data(nodes)\n");
    html.push_str("  .enter().append('g')\n");
    html.push_str("  .attr('class', 'node')\n");
    html.push_str("  .call(d3.drag()\n");
    html.push_str("    .on('start', dragstarted)\n");
    html.push_str("    .on('drag', dragged)\n");
    html.push_str("    .on('end', dragended));\n");

    // Shapes
    html.push_str("node.filter(d => d.shape === 'circle').append('circle')\n");
    html.push_str("  .attr('r', d => d.radius)\n");
    html.push_str("  .attr('class', d => d.hasGrad ? d.type : d.type + ' no-grad');\n");

    html.push_str("node.filter(d => d.shape === 'rect').append('rect')\n");
    html.push_str("  .attr('width', 40)\n");
    html.push_str("  .attr('height', 25)\n");
    html.push_str("  .attr('x', -20)\n");
    html.push_str("  .attr('y', -12.5)\n");
    html.push_str("  .attr('rx', 5)\n");
    html.push_str("  .attr('class', d => d.type);\n");

    // Scalar labels
    html.push_str("node.filter(d => d.shape === 'circle').append('text')\n");
    html.push_str("  .attr('dy', -30)\n");
    html.push_str("  .text(d => `val: ${d.value}`);\n");

    html.push_str("node.filter(d => d.shape === 'circle').append('text')\n");
    html.push_str("  .attr('dy', -15)\n");
    html.push_str("  .text(d => `grad: ${d.grad}`);\n");

    // Operation labels
    html.push_str("node.filter(d => d.shape === 'rect').append('text')\n");
    html.push_str("  .attr('dy', 4)\n");
    html.push_str("  .attr('font-size', '10px')\n");
    html.push_str("  .attr('font-weight', 'bold')\n");
    html.push_str("  .text(d => d.label);\n");

    // Legend
    html.push_str("const legend = svg.append('g')\n");
    html.push_str("  .attr('class', 'legend')\n");
    html.push_str("  .attr('transform', 'translate(20, 20)');\n");

    html.push_str(
        "legend.append('circle').attr('cx', 0).attr('cy', 0).attr('r', 8).attr('class', 'root');\n",
    );
    html.push_str("legend.append('text').attr('x', 15).attr('y', 5).text('Root Node');\n");

    html.push_str("legend.append('circle').attr('cx', 0).attr('cy', 25).attr('r', 8).attr('class', 'internal');\n");
    html.push_str("legend.append('text').attr('x', 15).attr('y', 30).text('Internal Node');\n");

    html.push_str("legend.append('circle').attr('cx', 0).attr('cy', 50).attr('r', 8).attr('class', 'leaf');\n");
    html.push_str("legend.append('text').attr('x', 15).attr('y', 55).text('Leaf Node');\n");

    html.push_str("legend.append('rect').attr('x', -4).attr('y', 71).attr('width', 16).attr('height', 8).attr('rx', 2).attr('class', 'operation');\n");
    html.push_str("legend.append('text').attr('x', 15).attr('y', 80).text('Operation Node');\n");

    html.push_str("legend.append('circle').attr('cx', 0).attr('cy', 95).attr('r', 8).attr('class', 'leaf no-grad');\n");
    html.push_str(
        "legend.append('text').attr('x', 15).attr('y', 100).text('Node without gradient');\n",
    );

    // Controls legend
    html.push_str("const instructions = svg.append('g')\n");
    html.push_str("  .attr('class', 'legend')\n");
    html.push_str("  .attr('transform', 'translate(20, 140)');\n");

    html.push_str("instructions.append('text').attr('x', 0).attr('y', 0).text('Controls:').style('font-weight', 'bold');\n");
    html.push_str("instructions.append('text').attr('x', 0).attr('y', 20).text('• Mouse wheel: Zoom in/out');\n");
    html.push_str("instructions.append('text').attr('x', 0).attr('y', 35).text('• Click + drag background: Pan');\n");
    html.push_str("instructions.append('text').attr('x', 0).attr('y', 50).text('• Drag nodes: Reposition');\n");

    // Tick callback
    html.push_str("simulation.on('tick', () => {\n");
    html.push_str("  link\n");
    html.push_str("    .attr('x1', d => d.source.x)\n");
    html.push_str("    .attr('y1', d => d.source.y)\n");
    html.push_str("    .attr('x2', d => d.target.x)\n");
    html.push_str("    .attr('y2', d => d.target.y);\n");

    html.push_str("  node\n");
    html.push_str("    .attr('transform', d => `translate(${d.x},${d.y})`);\n");
    html.push_str("});\n");

    // Drag handlers
    html.push_str("function dragstarted(event, d) {\n");
    html.push_str("  if (!event.active) simulation.alphaTarget(0.3).restart();\n");
    html.push_str("  d.fx = d.x;\n");
    html.push_str("  d.fy = d.y;\n");
    html.push_str("}\n");

    html.push_str("function dragged(event, d) {\n");
    html.push_str("  d.fx = event.x;\n");
    html.push_str("  d.fy = event.y;\n");
    html.push_str("}\n");

    html.push_str("function dragended(event, d) {\n");
    html.push_str("  if (!event.active) simulation.alphaTarget(0);\n");
    html.push_str("  d.fx = null;\n");
    html.push_str("  d.fy = null;\n");
    html.push_str("}\n");

    html.push_str("</script>\n");
    html.push_str("</body>\n</html>");

    // Persist HTML
    let mut output_file = File::create(file)?;
    output_file.write_all(html.as_bytes())?;

    println!("Graph visualization saved to: {}", file);
    Ok(())
}

/// Depth-first traversal that collects all nodes and edges reachable from
/// `current`.
fn collect_nodes_and_edges(
    current: &Scalar,
    visited: &mut HashSet<usize>,
    nodes: &mut Vec<NodeType>,
    edges: &mut Vec<(usize, usize)>,
    node_counter: &mut usize,
) {
    let current_id = current.get_arc_ptr();

    if visited.contains(&current_id) {
        return;
    }

    visited.insert(current_id);
    nodes.push(NodeType::Scalar {
        id: current_id,
        scalar: current.clone(),
    });

    // Traverse children
    let children = current.get_children();
    if !children.is_empty() {
        // Operation node
        *node_counter += 1000000; // Use a large offset to avoid ID conflicts
        let op_id = *node_counter;

        // Operation name
        let op_name = get_operation_name(current);

        nodes.push(NodeType::Operation { id: op_id, op_name });

        // Edge op→scalar
        edges.push((op_id, current_id));

        // Edges child→op
        for (child, _weight) in children.iter() {
            let child_id = child.get_arc_ptr();
            edges.push((child_id, op_id));
            collect_nodes_and_edges(&child, visited, nodes, edges, node_counter);
        }
    }
}

fn get_operation_name(scalar: &Scalar) -> String {
    scalar.get_operation_name()
}
